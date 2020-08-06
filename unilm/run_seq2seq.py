from absl import app, flags, logging
from glob import glob

import torch
import json
import sh
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import utils_seq2seq

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tokenization_unilm import UnilmTokenizer
from modeling_unilm import UnilmForSeq2Seq, UnilmConfig
from transformers import AdamW, set_seed

from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from sklearn.model_selection import train_test_split
from filelock import FileLock


flags.DEFINE_boolean('debug', True, '')
flags.DEFINE_boolean('mask_whole_word', False, '')

flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('epochs', 20, '')
flags.DEFINE_integer('num_workers', 4, '')
flags.DEFINE_integer('batch_size', 16, 'Total batch size for training.')
flags.DEFINE_integer('max_seq_length', 128, 'The maximum total input sequence length')
flags.DEFINE_integer('gpus', 1, '')
flags.DEFINE_integer('patience', 3, '')
flags.DEFINE_integer('max_position_embeddings', 512, 'max position embeddings')
flags.DEFINE_integer('max_pred', 20, 'max tokens of prediction')
flags.DEFINE_integer('skipgram_size', 1, 'the max size of ngram mask')

flags.DEFINE_float('learning_rate', 1e-5, '')
flags.DEFINE_float('weight_decay', 1e-2, '')
flags.DEFINE_float('adam_epsilon', 1e-8, '')
flags.DEFINE_float('test_size', 0.2, 'split data set size')
flags.DEFINE_float('label_smoothing', 0, 'The initial learning rate for Adam.')
flags.DEFINE_float('mask_prob', 0.2, 'number of prediction is sometimes less than max_pred when sequence is short')
flags.DEFINE_float('skipgram_prb', 0.0, 'prob of ngram mask')

flags.DEFINE_string('monitor', 'val_loss', '')
flags.DEFINE_string('metric_mode', 'min', '')

flags.DEFINE_string('model_name_or_path', 'bert-base-chinese', '')
flags.DEFINE_string('data_dir', os.path.join(os.getcwd(), 'data'), '')
flags.DEFINE_string('output_dir', os.path.join(os.getcwd(), 'output'), '')
flags.DEFINE_string('cache_dir', os.path.join(os.getcwd(), 'cache'), '')

FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

class QuestionAnswerGeneration(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.config = UnilmConfig.from_pretrained(
            FLAGS.model_name_or_path,
            max_position_embeddings=FLAGS.max_position_embeddings,
            label_smoothing=FLAGS.label_smoothing
        )

        self.tokenizer = UnilmTokenizer.from_pretrained(
            FLAGS.model_name_or_path
        )

        self.model = UnilmForSeq2Seq.from_pretrained(
            FLAGS.model_name_or_path,
            config=self.config
        )

    @staticmethod
    def _preprocess_qa_data(tokenizer):
        # 标注数据
        webqa_data = json.load(open(os.path.join(FLAGS.data_dir, 'WebQA.json')))
        sogou_data = json.load(open(os.path.join(FLAGS.data_dir, 'SogouQA.json')))
        train_data = webqa_data + sogou_data

        bi_uni_pipeline = [
            utils_seq2seq.Preprocess4Seq2seq(
                FLAGS.max_pred, 
                FLAGS.mask_prob, 
                list(tokenizer.vocab.keys()), 
                tokenizer.convert_tokens_to_ids, 
                FLAGS.max_seq_length, 
                mask_source_words=False, 
                skipgram_prb=FLAGS.skipgram_prb, 
                skipgram_size=FLAGS.skipgram_size, 
                mask_whole_word=FLAGS.mask_whole_word, 
                tokenizer=tokenizer
            )
        ]

        train_dataset = utils_seq2seq.Seq2SeqDataset(
            file_data=train_data, 
            batch_size=FLAGS.batch_size, 
            tokenizer=tokenizer, 
            max_len=FLAGS.max_seq_length, 
            bi_uni_pipeline=bi_uni_pipeline
        )

        return train_dataset

    def prepare_data(self):
        has_cache_files = False
        
        try:
            cache_dir = sh.ls(FLAGS.cache_dir)
            if 'train.set' in cache_dir and 'valid.set' in cache_dir:
                has_cache_files = True
        except Exception as e:
            logging.error(e)
            sh.mkdir(FLAGS.cache_dir)

        if not has_cache_files:
            data_list = self._preprocess_qa_data(self.tokenizer)
            train_list, valid_list = train_test_split(data_list, test_size=FLAGS.test_size, random_state=0)

            torch.save(train_list, os.path.join(FLAGS.cache_dir, 'train.set'))
            torch.save(valid_list, os.path.join(FLAGS.cache_dir, 'valid.set'))

    def forward(self, batch):
        input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch

        masked_lm_loss = self.model(
            input_ids, 
            segment_ids, 
            input_mask, 
            lm_label_ids,
            masked_pos=masked_pos, 
            masked_weights=masked_weights
        )

        return masked_lm_loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        out = {'val_loss': loss}

        return {**out, 'log': out}

    def train_dataloader(self):
        trainset_path = os.path.join(FLAGS.cache_dir, 'train.set')
        trainset_lock_path = trainset_path + '.lock'
        
        with FileLock(trainset_lock_path):
            train_set = torch.load(trainset_path)

        return DataLoader(
            train_set,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            collate_fn=utils_seq2seq.batch_list_to_batch_tensors
        )

    def val_dataloader(self):
        validset_path = os.path.join(FLAGS.cache_dir, 'valid.set')
        validset_lock_path = validset_path + '.lock'
        
        with FileLock(validset_lock_path):
            valid_set = torch.load(validset_path)

        return DataLoader(
            valid_set,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            collate_fn=utils_seq2seq.batch_list_to_batch_tensors
        )

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": FLAGS.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=FLAGS.learning_rate, eps=FLAGS.adam_epsilon)
        return optimizer
       

def main(argv):
    set_seed(FLAGS.seed)

    sh.rm('-r', '-f', FLAGS.output_dir)
    sh.mkdir(FLAGS.output_dir)

    model = QuestionAnswerGeneration()

    early_stop_callback = EarlyStopping(
        monitor=FLAGS.monitor,
        min_delta=0.0,
        patience=FLAGS.patience,
        verbose=True,
        mode=FLAGS.metric_mode,
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=FLAGS.output_dir,
        save_top_k=3,
        monitor=FLAGS.monitor,
        mode=FLAGS.metric_mode,
        prefix='qa_'
    )

    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(FLAGS.gpus if torch.cuda.is_available() else 0),
        distributed_backend='dp',
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger('logs/', name='qa', version=0),
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback
    )

    trainer.fit(model)

if __name__ == "__main__":
    app.run(main)
