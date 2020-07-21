from absl import app, flags, logging
from glob import glob

import torch
import json
import sh
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import GPT2LMHeadModel, BertTokenizer, GPT2Config
from transformers import AdamW
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from torchnlp.random import set_seed
from filelock import FileLock
from tqdm import tqdm

flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('train_mmi', False, '是否是训练mmi模型')
flags.DEFINE_boolean('use_pretrain', True, '是否使用预训练模型')

flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('epochs', 20, '')
flags.DEFINE_integer('num_workers', 4, '')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('gpus', 1, '')
flags.DEFINE_integer('patience', 3, '')

flags.DEFINE_float('learning_rate', 1e-5, '学习率')
flags.DEFINE_float('weight_decay', 1e-2, '')
flags.DEFINE_float('adam_epsilon', 1e-8, '')
flags.DEFINE_float('test_size', 0.2, '')

flags.DEFINE_string('monitor', 'val_loss', '')
flags.DEFINE_string('metric_mode', 'min', '')

flags.DEFINE_string('model_config', os.path.join(os.getcwd(), 'config', 'model_config_dialogue_small.json'), 'config文件')
flags.DEFINE_string('vocab_path', os.path.join(os.getcwd(), 'vocabulary', 'vocab_small.txt'), 'vocabulary文件')
flags.DEFINE_string('train_raw_path', os.path.join(os.getcwd(), 'data', 'train.txt'), '训练数据')

flags.DEFINE_string('dialogue_model_path', os.path.join(os.getcwd(), 'models/dialogue_model'), 'dialogue的预训练模型')
flags.DEFINE_string('mmi_model_path', os.path.join(os.getcwd(), 'models/mmi_model'), 'mmi的预训练模型')
flags.DEFINE_string('output_dir', os.path.join(os.getcwd(), 'output'), '训练好的mmi的模型目录')
flags.DEFINE_string('cache_dir', os.path.join(os.getcwd(), 'cache'), '处理完的数据的目录')

FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

class ChitChat(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.config = GPT2Config.from_json_file(
            FLAGS.model_config
        )

        self.tokenizer = BertTokenizer(
            vocab_file=FLAGS.vocab_path
        )

        if FLAGS.train_mmi and FLAGS.use_pretrain:
            self.model = GPT2LMHeadModel.from_pretrained(
                FLAGS.mmi_model_path,
                config=self.config
            )
        elif not FLAGS.train_mmi and FLAGS.use_pretrain:
            self.model = GPT2LMHeadModel.from_pretrained(
                FLAGS.dialogue_model_path,
                config=self.config
            )
        else:
            self.model = GPT2LMHeadModel(
                config=self.config
            )

        self.n_ctx = self.model.config.to_dict().get("n_ctx")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("[PAD]")

    def prepare_data(self): # 数据的预处理转换
        has_cache_files = False
        has_mmi_cache_files = False
        
        try:
            cache_dir = sh.ls(FLAGS.cache_dir)
            if 'train.set' in cache_dir and 'valid.set' in cache_dir:
                has_cache_files = True
            if 'train_mmi.set' in cache_dir and 'valid_mmi.set' in cache_dir:
                has_mmi_cache_files = True
        except Exception as e:
            logging.error(e)
            sh.mkdir(FLAGS.cache_dir)

        if not has_cache_files and not FLAGS.train_mmi:
            data_list = self._preprocess_raw_data(self.tokenizer, self.n_ctx)
            train_list, valid_list = train_test_split(data_list, test_size=FLAGS.test_size, random_state=0)

            train_input_ids, train_attention_mask = self._seq_padding(self.tokenizer, train_list)
            valid_input_ids, valid_attention_mask = self._seq_padding(self.tokenizer, valid_list)

            dialogue_train_set = TensorDataset(train_input_ids, train_attention_mask)
            dialogue_valid_set = TensorDataset(valid_input_ids, valid_attention_mask)

            torch.save(dialogue_train_set, os.path.join(FLAGS.cache_dir, 'train.set'))
            torch.save(dialogue_valid_set, os.path.join(FLAGS.cache_dir, 'valid.set'))
        
        if not has_mmi_cache_files and FLAGS.train_mmi:
            data_list = self._preprocess_mmi_raw_data(self.tokenizer, self.n_ctx)
            train_list, valid_list = train_test_split(data_list, test_size=FLAGS.test_size, random_state=0)

            train_input_ids, train_attention_mask = self._seq_padding(self.tokenizer, train_list)
            valid_input_ids, valid_attention_mask = self._seq_padding(self.tokenizer, valid_list)

            dialogue_train_set = TensorDataset(train_input_ids, train_attention_mask)
            dialogue_valid_set = TensorDataset(valid_input_ids, valid_attention_mask)

            torch.save(dialogue_train_set, os.path.join(FLAGS.cache_dir, 'train_mmi.set'))
            torch.save(dialogue_valid_set, os.path.join(FLAGS.cache_dir, 'valid_mmi.set'))

    @staticmethod
    def _preprocess_raw_data(tokenizer, n_ctx):
        with open(FLAGS.train_raw_path, 'rb') as f:
            data = f.read().decode("utf-8")
        if "\r\n" in data:
            train_data = data.split("\r\n\r\n")
        else:
            train_data = data.split("\n\n")
        
        processed_data = []
        
        for dialogue in tqdm(train_data):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            for utterance in utterances:
                dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            dialogue_ids = dialogue_ids[:n_ctx]
            processed_data.append(dialogue_ids)
        
        return processed_data
    
    @staticmethod
    def _preprocess_mmi_raw_data(tokenizer, n_ctx):
        with open(FLAGS.train_raw_path, 'rb') as f:
            data = f.read().decode("utf-8")
        if "\r\n" in data:
            train_data = data.split("\r\n\r\n")
        else:
            train_data = data.split("\n\n")
        
        processed_data = []
        
        for dialogue in tqdm(train_data):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")
            dialogue_ids = [tokenizer.cls_token_id]  # 每个dialogue以[CLS]开头
            
            for utterance in reversed(utterances):
                dialogue_ids.extend([tokenizer.convert_tokens_to_ids(word) for word in utterance])
                dialogue_ids.append(tokenizer.sep_token_id)  # 每个utterance之后添加[SEP]，表示utterance结束
            # 对超过n_ctx的长度进行截断,否则GPT2模型会报错
            dialogue_ids = dialogue_ids[:n_ctx]
            processed_data.append(dialogue_ids)
        
        return processed_data

    @staticmethod
    def _seq_padding(tokenizer, X):
        pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

        if len(X) <= 1:
            attention_mask = torch.tensor([[1]*len(x) for x in X], dtype=torch.long)
            return torch.tensor(X, dtype=torch.long), attention_mask
        L = [len(x) for x in X]
        ML = max(L)

        attention_mask = torch.tensor([[1]*len(x) + [0]*(ML - len(x)) if len(x) < ML else [1]*len(x) for x in X], dtype=torch.long)
        X = torch.tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X], dtype=torch.long)
        
        return X, attention_mask

    def _calculate_loss_and_accuracy(self, outputs, labels):
        """
        计算非pad_id的平均loss和准确率
        :param outputs:
        :param labels:
        :return:
        """
        logits = outputs[0] # 每个token用来预测下一个token的prediction_score,维度:[batch_size,token_len,voca_size]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss(ignore_index=self.pad_id, reduction='sum')  # 忽略pad_id的loss,并对所有的非pad_id的loss进行求和
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        _, preds = shift_logits.max(dim=-1)  # preds表示对应的prediction_score预测出的token在voca中的id。维度为[batch_size,token_len]

        # 对非pad_id的token的loss进行求平均，且计算出预测的准确率
        not_ignore = shift_labels.ne(self.pad_id)  # 进行非运算，返回一个tensor，若targets_view的第i个位置为pad_id，则置为0，否则为1
        num_targets = not_ignore.long().sum().item()  # 计算target中的非pad_id的数量

        correct = (shift_labels == preds) & not_ignore  # 计算model预测正确的token的个数，排除pad的tokne
        correct = correct.float().sum()

        accuracy = correct / num_targets
        loss = loss / num_targets
        return loss, accuracy

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1])
        loss, accuracy = self._calculate_loss_and_accuracy(outputs, batch[0])
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1])
        loss, accuracy = self._calculate_loss_and_accuracy(outputs, batch[0])
        return {'loss': loss, 'acc': accuracy}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = torch.stack([x['acc'] for x in outputs]).mean()
        out = {'val_loss': loss, 'val_acc': acc}

        return {**out, 'log': out}

    def train_dataloader(self):
        if FLAGS.train_mmi:
            trainset_path = os.path.join(FLAGS.cache_dir, 'train_mmi.set')
            trainset_lock_path = trainset_path + '.lock'
        else:
            trainset_path = os.path.join(FLAGS.cache_dir, 'train.set')
            trainset_lock_path = trainset_path + '.lock'
        
        with FileLock(trainset_lock_path):
            nlu_train_set = torch.load(trainset_path)

        return DataLoader(
            nlu_train_set,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers
        )

    def val_dataloader(self):
        if FLAGS.train_mmi:
            validset_path = os.path.join(FLAGS.cache_dir, 'valid_mmi.set')
            validset_lock_path = validset_path + '.lock'
        else:
            validset_path = os.path.join(FLAGS.cache_dir, 'valid.set')
            validset_lock_path = validset_path + '.lock'
        
        with FileLock(validset_lock_path):
            nlu_valid_set = torch.load(validset_path)

        return DataLoader(
            nlu_valid_set,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers
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

    if not os.path.exists(FLAGS.output_dir):
        sh.mkdir(FLAGS.output_dir)

    model = ChitChat()

    early_stop_callback = EarlyStopping(
        monitor=FLAGS.monitor,
        min_delta=0.0,
        patience=FLAGS.patience,
        verbose=True,
        mode=FLAGS.metric_mode,
    )

    if FLAGS.train_mmi:
        checkpoint_callback = ModelCheckpoint(
            filepath=FLAGS.output_dir,
            save_top_k=3,
            monitor=FLAGS.monitor,
            mode=FLAGS.metric_mode,
            prefix='mmi_model_'
        )
        logger = pl.loggers.TensorBoardLogger('logs/', name='mmi', version=0)
    else:
        checkpoint_callback = ModelCheckpoint(
            filepath=FLAGS.output_dir,
            save_top_k=3,
            monitor=FLAGS.monitor,
            mode=FLAGS.metric_mode,
            prefix='dialogue_model_'
        )
        logger = pl.loggers.TensorBoardLogger('logs/', name='dialogue', version=0)

    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(FLAGS.gpus if torch.cuda.is_available() else 0),
        distributed_backend='dp',
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback
    )

    trainer.fit(model)
    
if __name__ == "__main__":
    app.run(main)
