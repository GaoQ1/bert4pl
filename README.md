# BERT for Python-Lightning
## BERT for Python-Lightning
- [优雅の使用transformer系列之text-classification](https://www.jianshu.com/p/37346c8873bb)

## Introduction
这个repo会持续放基于transformer和pytorch-lightning的一些功能实现，也欢迎大家contribute.

## Running by command
### install packages
 - python >= 3.6
 - pytorch
```
pip install -r requirements.txt
```
下载依赖package

## Run_Classification
### train model
```
python run_classification_pl.py
```
关于bert预训练模型，可以通过指定'--model_name_or_path=YOUR DOWNLOAD BERT MODEL PATH'来避免去外网下载，这里的训练数据来自[rasa-nlu-benchmark](https://github.com/nghuyong/rasa-nlu-benchmark)

### run model and provide a service
```
python run_classification_pl.py --do_train=False --do_predict=True
```

### test by http server
`http://localhost:5000/parse` post请求，请求参数例如：
```
["糖醋排骨怎么做啊？"]
```
当然也可以使用postman去请求调用

## Run_NER
### train model
```
python run_ner_pl.py
```
关于bert预训练模型，可以通过指定'--model_name_or_path=YOUR DOWNLOAD BERT MODEL PATH'来避免去外网下载

### run model and provide a service
```
python run_ner_pl.py --do_train=False --do_predict=True
```

## Run_GPT2_Chitchat
### train model
```
cd gpt2-chichat-pl
python train.py
```

### run model and provide a service
```
python run_interact.py --train_mmi=True
python run_interact.py --train_mmi=False
```

## 未完待续...
