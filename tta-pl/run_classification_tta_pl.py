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
from model_customize.modeling_tta import BertForTtaNlu
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, set_seed
from torch.utils.data import DataLoader, TensorDataset
from rasa.nlu.training_data.loading import load_data
from sklearn.model_selection import train_test_split
from filelock import FileLock


flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('do_train', True, '')
flags.DEFINE_boolean('do_predict', False, '')

flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_integer('epochs', 20, '')
flags.DEFINE_integer('num_workers', 4, '')
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('max_seq_length', 256, '')
flags.DEFINE_integer('intent_ranking_length', 5, '')
flags.DEFINE_integer('gpus', 1, '')
flags.DEFINE_integer('patience', 3, '')

flags.DEFINE_float('learning_rate', 1e-5, '')
flags.DEFINE_float('weight_decay', 1e-2, '')
flags.DEFINE_float('adam_epsilon', 1e-8, '')

flags.DEFINE_string('monitor', 'val_loss', '')
flags.DEFINE_string('metric_mode', 'min', '')

flags.DEFINE_string('model_name_or_path', 'bert-base-chinese', '')
flags.DEFINE_string('data_dir', os.path.join(os.getcwd(), 'data'), '')
flags.DEFINE_string('cache_dir', os.path.join(os.getcwd(), 'cache'), '')
flags.DEFINE_string('logs_dir', os.path.join(os.getcwd(), 'logs'), '')
flags.DEFINE_string('output_dir', os.path.join(os.getcwd(), 'output'), '')

FLAGS = flags.FLAGS

class NluClassifier(pl.LightningModule):
    def __init__(self, id2class=None, intent_examples=None):
        super().__init__()
        self.id2class = id2class
        self.class2id = {j: i for i, j in self.id2class.items()}
        self.intent_examples = intent_examples

        self.config = AutoConfig.from_pretrained(
            FLAGS.model_name_or_path,
            num_labels=len(id2class)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            FLAGS.model_name_or_path
        )

        self.model = BertForTtaNlu.from_pretrained(
            FLAGS.model_name_or_path,
            config=self.config
        )

        self.loss = nn.CrossEntropyLoss()

    def prepare_data(self): # 数据的预处理转换
        has_cache_files = False
        
        try:
            cache_dir = sh.ls(FLAGS.cache_dir)
            if 'train.set' in cache_dir and 'valid.set' in cache_dir:
                has_cache_files = True
        except Exception as e:
            logging.error(e)
            sh.mkdir(FLAGS.cache_dir)

        if not has_cache_files:
            X, Y = [], []

            for msg in self.intent_examples:
                X.append(msg.text)
                Y.append(self.class2id[msg.get('intent')])
            
            X, _ = self._convert_text_to_ids(self.tokenizer, X, FLAGS.max_seq_length)
            train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.3, random_state=0)

            train_input_ids, train_attention_mask = self._seq_padding(self.tokenizer, train_x)
            train_labels = torch.tensor(train_y, dtype=torch.long)

            valid_input_ids, valid_attention_mask = self._seq_padding(self.tokenizer, valid_x)
            valid_labels = torch.tensor(valid_y, dtype=torch.long)

            nlu_train_set = TensorDataset(train_input_ids, train_attention_mask, train_labels)
            nlu_valid_set = TensorDataset(valid_input_ids, valid_attention_mask, valid_labels)

            torch.save(nlu_train_set, os.path.join(FLAGS.cache_dir, 'train.set'))
            torch.save(nlu_valid_set, os.path.join(FLAGS.cache_dir, 'valid.set'))

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

        return outputs

    def predict(self, text):
        if self.training:
            self.eval()

        with torch.no_grad():
            text_to_id, _ = self._convert_text_to_ids(self.tokenizer, text, FLAGS.max_seq_length)
            input_ids, attention_mask = self._seq_padding(self.tokenizer, text_to_id)
            outputs = self(input_ids, attention_mask)
            probs = F.softmax(outputs[0], dim=-1)

            intent_ranking = [
                {
                    "intent": self.id2class[idx], 
                    "confidence": float(score.item())
                } for idx, score in enumerate(probs[0])
            ]

            intent_ranking = sorted(intent_ranking,
                                    key=lambda s: s['confidence'],
                                    reverse=True)
            intent_ranking = intent_ranking[:FLAGS.intent_ranking_length]

            intent = intent_ranking[0]

            predict_result = {
                "text": text[0],
                **intent,
                "intent_ranking": intent_ranking
            }

        return predict_result

    @staticmethod
    def _convert_text_to_ids(tokenizer, text, max_len=100):
        if isinstance(text, str):
            tokenizer_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True)
            input_ids = tokenizer_text["input_ids"]
            token_type_ids = tokenizer_text["token_type_ids"]
        elif isinstance(text, list):
            input_ids = []
            token_type_ids = []
            for t in text:
                tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True)
                input_ids.append(tokenized_text["input_ids"])
                token_type_ids.append(tokenized_text["token_type_ids"])
        else:
            print("Unexpected input")
        return input_ids, token_type_ids
    
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

    def training_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1], batch[2])
        loss = outputs[0].mean()

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        outputs = self(batch[0], batch[1], batch[2])
        logits, loss = outputs[0], outputs[1].mean()

        acc = (logits.argmax(-1) == batch[2]).float()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}

        return {**out, 'log': out}

    def train_dataloader(self):
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


def read_nlu_data():
    try:
        cache_dir = sh.ls(FLAGS.cache_dir)
        if 'id2class.set' in cache_dir and 'intent_examples.set' in cache_dir:
            id2class_path = os.path.join(FLAGS.cache_dir, 'id2class.set')
            id2class_lock_path = id2class_path + '.lock'
            intent_examples_path = os.path.join(FLAGS.cache_dir, 'intent_examples.set')
            intent_examples_lock_path = intent_examples_path + '.lock'
            
            with FileLock(id2class_lock_path):
                id2class = torch.load(id2class_path)

            with FileLock(intent_examples_lock_path):
                intent_examples = torch.load(intent_examples_path)

            return id2class, intent_examples
    except Exception as e:
        logging.error(e)
        sh.mkdir(FLAGS.cache_dir)

    data = load_data(FLAGS.data_dir, 'zh')
    id2class = dict(enumerate(data.intents))
    intent_examples = data.intent_examples

    torch.save(id2class, os.path.join(FLAGS.cache_dir, 'id2class.set'))
    torch.save(intent_examples, os.path.join(FLAGS.cache_dir, 'intent_examples.set'))

    return id2class, intent_examples
       

def main(argv):
    if not os.path.exists(FLAGS.logs_dir):
        os.makedirs(FLAGS.logs_dir)

    set_seed(FLAGS.seed)
    id2class, intent_examples = read_nlu_data()

    if FLAGS.do_train:
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)

        model = NluClassifier(id2class, intent_examples)

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
            prefix='nlu_tta_'
        )

        trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(FLAGS.gpus if torch.cuda.is_available() else 0),
            distributed_backend='dp',
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            logger=pl.loggers.TensorBoardLogger('logs/', name='nlu_tta', version=0),
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback
        )

        trainer.fit(model)

    if FLAGS.do_predict:
        from sanic import Sanic, response
        server = Sanic()

        checkpoints = list(sorted(glob(os.path.join(FLAGS.output_dir, "nlu_tta_*.ckpt"), recursive=True)))
        model = NluClassifier.load_from_checkpoint(
            checkpoint_path= checkpoints[-1],
            id2class= id2class, 
            intent_examples= intent_examples
        )
        model.eval()
        model.freeze()

        @server.route("/parse", methods=['POST'])
        async def parse(request):
            texts = request.json
            prediction = model.predict(texts)
            return response.json(prediction)

        server.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    app.run(main)
