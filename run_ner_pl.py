from absl import app, flags, logging
from glob import glob

import torch
import json
import sh
import os
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from seqeval.metrics import f1_score, precision_score, recall_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
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
flags.DEFINE_integer('max_seq_length', 128, '')
flags.DEFINE_integer('intent_ranking_length', 5, '')
flags.DEFINE_integer('gpus', 1, '')
flags.DEFINE_integer('patience', 3, '')
flags.DEFINE_integer('pad_token_label_id', -100, '')

flags.DEFINE_float('learning_rate', 1e-5, '')
flags.DEFINE_float('weight_decay', 1e-2, '')
flags.DEFINE_float('adam_epsilon', 1e-8, '')
flags.DEFINE_float('test_size', 0.3, '')

flags.DEFINE_string('monitor', 'val_loss', '')
flags.DEFINE_string('metric_mode', 'min', '')

flags.DEFINE_string('model_name_or_path', 'bert-base-chinese', '')
flags.DEFINE_string('data_dir', os.path.join(os.getcwd(), 'data'), '')
flags.DEFINE_string('output_dir', os.path.join(os.getcwd(), 'output'), '')
flags.DEFINE_string('cache_dir', os.path.join(os.getcwd(), 'cache'), '')

FLAGS = flags.FLAGS

sh.rm('-r', '-f', 'logs')
sh.mkdir('logs')

class NerClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.id2entity, self.entity_examples = self._read_nlu_data()

        self.entity2id = {j: i for i, j in self.id2entity.items()}

        self.config = AutoConfig.from_pretrained(
            FLAGS.model_name_or_path,
            num_labels=len(self.id2entity)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            FLAGS.model_name_or_path
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            FLAGS.model_name_or_path,
            config=self.config
        )

        self.loss = nn.CrossEntropyLoss()

    def _read_nlu_data(self):
        try:
            cache_dir = sh.ls(FLAGS.cache_dir)
            if 'id2entity.set' in cache_dir and 'entity_examples.set' in cache_dir:
                id2entity_path = os.path.join(FLAGS.cache_dir, 'id2entity.set')
                id2entity_lock_path = id2entity_path + '.lock'

                entity_examples_path = os.path.join(FLAGS.cache_dir, 'entity_examples.set')
                entity_examples_lock_path = entity_examples_path + '.lock'
                
                with FileLock(id2entity_lock_path):
                    id2entity = torch.load(id2entity_path)

                with FileLock(entity_examples_lock_path):
                    entity_examples = torch.load(entity_examples_path)

                return id2entity, entity_examples
        except Exception as e:
            logging.error(e)
            sh.mkdir(FLAGS.cache_dir)
        
        data = load_data(FLAGS.data_dir, 'zh')
        entities, entity_examples = data.entities, data.entity_examples
        entity_lists, entity_examples_cooked = ['O'], []

        for example in entity_examples:
            entity_examples_cooked.append(self._predata(example.text, example.get("entities", [])))

        for entity in entities:
            for tag in ['B', 'I']:
                entity_lists.append(tag + '-' + entity)

        id2entity = dict(enumerate(entity_lists))

        torch.save(id2entity, os.path.join(FLAGS.cache_dir, 'id2entity.set'))
        torch.save(entity_examples_cooked, os.path.join(FLAGS.cache_dir, 'entity_examples.set'))

        return id2entity, entity_examples_cooked

    @staticmethod
    def _predata(text, entity_offsets):
        value = 'O'
        text = text.rstrip()
        bilou = [value for _ in text]

        cooked_data = []

        for entity_offset in entity_offsets:
            start, end, entity = entity_offset.get('start', None), entity_offset.get('end', None), entity_offset.get('entity', None)
            if start is not None and end is not None:
                bilou[start] = 'B-' + entity
                for i in range(start+1, end):
                    bilou[i] = 'I-' + entity

        for index, achar in enumerate(text):
            if achar.strip():
                temp = []
                temp.append(achar)
                temp.append(bilou[index])
                cooked_data.append(temp)
            else:
                continue

        return cooked_data

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

            for entity_example in self.entity_examples:
                item_x, item_y = [], []
                for item in entity_example:
                    item_x.append(item[0])
                    item_y.append(self.entity2id[item[1]])
                X.append(item_x)
                Y.append(item_y)

            train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=FLAGS.test_size, random_state=0)

            train_input_ids = self._convert_text_to_ids(self.tokenizer, train_x)
            valid_input_ids = self._convert_text_to_ids(self.tokenizer, valid_x)

            train_input_ids, train_labels, train_attention_masks = self._convert_example_to_features(self.tokenizer, train_input_ids, train_y, FLAGS.pad_token_label_id)
            valid_input_ids, valid_labels, valid_attention_masks = self._convert_example_to_features(self.tokenizer, valid_input_ids, valid_y, FLAGS.pad_token_label_id)

            nlu_train_set = TensorDataset(train_input_ids, train_attention_masks, train_labels)
            nlu_valid_set = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)

            torch.save(nlu_train_set, os.path.join(FLAGS.cache_dir, 'train.set'))
            torch.save(nlu_valid_set, os.path.join(FLAGS.cache_dir, 'valid.set'))

    def forward(self, **inputs):
        return self.model(**inputs)

    def predict(self, text):
        if self.training:
            self.eval()

        with torch.no_grad():
            input_ids, attention_masks = [], []
            for t in text:
                text_encode = self.tokenizer.encode_plus(t, max_length=FLAGS.max_seq_length, pad_to_max_length=True)
                input_ids.append(text_encode["input_ids"])
                attention_masks.append(text_encode["attention_mask"])

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)
            inputs = {"input_ids": input_ids, "attention_mask": attention_masks}
            outputs = self(**inputs)

        pred_lists = self._predict_outputs(outputs)
        result = []

        for t, tags in zip(text, pred_lists):
            result.append(self._result_to_json(t, tags[1: len(t) + 1]))
        
        return result
    
    def _predict_outputs(self, outputs):
        preds = np.argmax(outputs[0], axis=2)
        preds_list = [[] for _ in range(preds.shape[0])]
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                preds_list[i].append(self.id2entity[preds[i][j].item()])
        return preds_list

    @staticmethod
    def _result_to_json(string, tags):
        item = {
            "string": string,
            "entities": []
        }
        entity_name = ""
        entity_start = 0
        idx = 0

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                item["entities"].append(
                    {"value": char, "start": idx, "end": idx+1, "entity": tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append(
                    {"value": entity_name,
                    "start": entity_start,
                    "end": idx + 1,
                    "entity": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
        return item 

    @staticmethod
    def _convert_text_to_ids(tokenizer, text):
        if isinstance(text, list):
            input_ids = []
            for t in text:
                tokenized_text = tokenizer.encode_plus(t, add_special_tokens=False)
                input_ids.append(tokenized_text["input_ids"])
        else:
            print("Unexpected input")
        return input_ids

    @staticmethod
    def _convert_example_to_features(tokenizer, x, y, maxlen=128, pad_token_label_id=-100):
        assert len(x) == len(y)
        cls_id = tokenizer.convert_tokens_to_ids("[CLS]")
        pad_id = tokenizer.convert_tokens_to_ids("[PAD]")

        x_features, y_features, attention_masks = [], [], []

        for x_item, y_item in zip(x, y):
            x_feature, y_feature, attention_mask = [], [], []

            x_feature += [cls_id]
            y_feature += [pad_token_label_id]

            x_feature += x_item
            y_feature += y_item

            x_feature += [pad_id]
            y_feature += [pad_token_label_id]

            attention_mask = [1] * len(x_feature)

            if len(x_feature) > maxlen:
                x_feature = x_feature[:maxlen]
                y_feature = y_feature[:maxlen]
                attention_mask = attention_mask[:maxlen]
            else:
                x_feature = x_feature + [pad_id] * (maxlen- len(x_feature))
                y_feature = y_feature + [pad_token_label_id] * (maxlen- len(y_feature))
                attention_mask = attention_mask + [0] * (maxlen- len(attention_mask))

            x_features.append(x_feature)        
            y_features.append(y_feature)            
            attention_masks.append(attention_mask)        

        x_features = torch.tensor(x_features, dtype=torch.long)
        y_features = torch.tensor(y_features, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        
        return x_features, y_features, attention_masks
    
    def training_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        outputs = self(**inputs)
        loss = outputs[0]

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
        outputs = self(**inputs)
        loss, logits = outputs[:2]

        preds = logits.detach().cpu().numpy()
        out_label_ids = inputs["labels"].detach().cpu().numpy()

        return {'val_loss': loss.detach().cpu(), "pred": preds, "target": out_label_ids}

    def _eval_end(self, outputs):
        "Evaluation called for both Val and Test"
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        preds = np.concatenate([x["pred"] for x in outputs], axis=0)
        preds = np.argmax(preds, axis=2)
        out_label_ids = np.concatenate([x["target"] for x in outputs], axis=0)

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != FLAGS.pad_token_label_id:
                    out_label_list[i].append(self.id2entity[out_label_ids[i][j]])
                    preds_list[i].append(self.id2entity[preds[i][j]])

        results = {
            "val_loss": val_loss_mean,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

        ret = {k: v for k, v in results.items()}
        ret["log"] = results
        return ret, preds_list, out_label_list

    def validation_epoch_end(self, outputs):
        ret, preds, targets = self._eval_end(outputs)
        logs = ret["log"]
        return {"val_loss": logs["val_loss"], "log": logs, "progress_bar": logs}

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
       

def main(argv):
    set_seed(FLAGS.seed)

    if FLAGS.do_train:
        sh.rm('-r', '-f', FLAGS.output_dir)
        sh.mkdir(FLAGS.output_dir)

        model = NerClassifier()

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
            prefix='ner_'
        )

        trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(FLAGS.gpus if torch.cuda.is_available() else 0),
            distributed_backend='dp',
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            logger=pl.loggers.TensorBoardLogger('logs/', name='ner', version=0),
            checkpoint_callback=checkpoint_callback,
            early_stop_callback=early_stop_callback
        )

        trainer.fit(model)

    if FLAGS.do_predict:
        checkpoints = list(sorted(glob(os.path.join(FLAGS.output_dir, "*.ckpt"), recursive=True)))
        model = NerClassifier.load_from_checkpoint(
            checkpoint_path= checkpoints[-1]
        )
        model.eval()
        model.freeze()

        while True:
            text = input("输入：")
            prediction = model.predict([text])
            print("ner的结果是：", prediction)

if __name__ == "__main__":
    app.run(main)
