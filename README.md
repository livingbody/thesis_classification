# 一、学术论文分类挑战赛

赛题链接：[http://challenge.xfyun.cn/h5/invite?invitaCode=8zCBfV](http://challenge.xfyun.cn/h5/invite?invitaCode=8zCBfV)

aistudio连接 [https://aistudio.baidu.com/aistudio/projectdetail/2191235](https://aistudio.baidu.com/aistudio/projectdetail/2191235)


## 1.赛事背景
随着人工智能技术不断发展，每周都有非常多的论文公开发布。现如今对论文进行分类逐渐成为非常现实的问题，这也是研究人员和研究机构每天都面临的问题。现在希望选手能构建一个论文分类模型。

## 2.赛事任务
本次赛题希望参赛选手利用论文信息：论文id、标题、摘要，划分论文具体类别。

## 赛题样例（使用\t分隔）：

paperid：9821

title：Calculation of prompt diphoton production cross sections at Tevatron and LHC energies

abstract：A fully differential calculation in perturbative quantum chromodynamics is presented for the production of massive photon pairs at hadron colliders. All next-to-leading order perturbative contributions from quark-antiquark, gluon-(anti)quark, and gluon-gluon subprocesses are included, as well as all-orders resummation of initial-state gluon radiation valid at next-to-next-to-leading logarithmic accuracy.

categories：hep-ph

## 3.预测结果文件详细说明：

1) 以csv格式提交，编码为UTF-8，第一行为表头；

2) 提交前请确保预测结果的格式与sample_submit.csv中的格式一致。具体格式如下：

paperid,categories

test_00000,cs.CV

test_00001,cs.DC

test_00002,cs.AI

test_00003,cs.NI

test_00004,cs.SE

#  二、数据处理

## 1 升级paddlenlp
```
Found existing installation: paddlenlp 2.0.1
    Uninstalling paddlenlp-2.0.1:
      Successfully uninstalled paddlenlp-2.0.1
Successfully installed paddlenlp-2.0.5
```


```python
!pip install -U paddlenlp
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already up-to-date: paddlenlp in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (2.0.6)
    Requirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.9.0)
    Requirement already satisfied, skipping upgrade: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.4.4)
    Requirement already satisfied, skipping upgrade: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.70.11.1)
    Requirement already satisfied, skipping upgrade: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.42.1)
    Requirement already satisfied, skipping upgrade: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (4.1.0)
    Requirement already satisfied, skipping upgrade: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (1.2.2)
    Requirement already satisfied, skipping upgrade: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.20.3)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.8.2)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (7.1.2)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.21.0)
    Requirement already satisfied, skipping upgrade: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.5)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.0.0)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.14.0)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.22.0)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.15.0)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.8.53)
    Requirement already satisfied, skipping upgrade: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.2.3)
    Requirement already satisfied, skipping upgrade: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp) (0.3.3)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp) (0.24.2)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.6.1)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.6.0)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.23)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (7.0)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (2.10.1)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (0.16.0)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.4.10)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (5.1.2)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (16.7.9)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (2.0.1)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.0)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.4)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl->paddlenlp) (2019.3)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (3.0.4)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2019.9.11)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2.8)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (1.25.6)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (3.9.9)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (0.18.0)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (2.4.2)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (2.1.0)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (1.6.3)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (0.14.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp) (0.6.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl->paddlenlp) (56.2.0)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp) (7.2.0)



```python
import pandas as pd
from paddlenlp.datasets import load_dataset
import paddlenlp as ppnlp
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import  convert_example, create_dataloader
import os
import numpy as np
import paddle
import paddle.nn.functional as F
```

## 2.解压缩


```python
# 解压缩
# !unzip -oq /home/aistudio/data/data100202/Datawhale_学术论文分类_数据集.zip -d dataset
# !rm dataset/__MACOSX/ -rf
# !unzip -oq /home/aistudio/dataset/Datawhale_学术论文分类_数据集/test.csv.zip -d dataset/
# !unzip -oq /home/aistudio/dataset/Datawhale_学术论文分类_数据集/train.csv.zip -d dataset/
```

## 3.数据查看


```python
# 提交格式
!head dataset/Datawhale_学术论文分类_数据集/sample_submit.csv
```

    paperid,categories
    test_00000,cs.CV
    test_00001,cs.CV
    test_00002,cs.CV
    test_00003,cs.CV
    test_00004,cs.CV
    test_00005,cs.CV
    test_00006,cs.CV
    test_00007,cs.CV
    test_00008,cs.CV



```python
# train数据格式
!head -n20 dataset/train.csv
```

    paperid	title	abstract	categories
    train_00000	"Hard but Robust, Easy but Sensitive: How Encoder and Decoder Perform in
      Neural Machine Translation"	"  Neural machine translation (NMT) typically adopts the encoder-decoder
    framework. A good understanding of the characteristics and functionalities of
    the encoder and decoder can help to explain the pros and cons of the framework,
    and design better models for NMT. In this work, we conduct an empirical study
    on the encoder and the decoder in NMT, taking Transformer as an example. We
    find that 1) the decoder handles an easier task than the encoder in NMT, 2) the
    decoder is more sensitive to the input noise than the encoder, and 3) the
    preceding words/tokens in the decoder provide strong conditional information,
    which accounts for the two observations above. We hope those observations can
    shed light on the characteristics of the encoder and decoder and inspire future
    research on NMT.
    "	cs.CL
    train_00001	An Easy-to-use Real-world Multi-objective Optimization Problem Suite	"  Although synthetic test problems are widely used for the performance
    assessment of evolutionary multi-objective optimization algorithms, they are
    likely to include unrealistic properties which may lead to
    overestimation/underestimation. To address this issue, we present a
    multi-objective optimization problem suite consisting of 16 bound-constrained
    real-world problems. The problem suite includes various problems in terms of



```python
# test数据格式
!head dataset/test.csv
```

    paperid	title	abstract
    test_00000	"Analyzing 2.3 Million Maven Dependencies to Reveal an Essential Core in
      APIs"	"  This paper addresses the following question: does a small, essential, core
    set of API members emerges from the actual usage of the API by client
    applications? To investigate this question, we study the 99 most popular
    libraries available in Maven Central and the 865,560 client programs that
    declare dependencies towards them, summing up to 2.3M dependencies. Our key
    findings are as follows: 43.5% of the dependencies declared by the clients are
    not used in the bytecode; all APIs contain a large part of rarely used types
    and a few frequently used types, and the ratio varies according to the nature


## 4.自定义read方法


```python
import pandas as pd

train = pd.read_csv('dataset/train.csv', sep='\t')
test = pd.read_csv('dataset/test.csv', sep='\t')
sub = pd.read_csv('dataset/Datawhale_学术论文分类_数据集/sample_submit.csv')
# 拼接title与abstract
train['text'] = train['title'] + ' ' + train['abstract']

```


```python
label_id2cate = dict(enumerate(train.categories.unique()))
label_cate2id = {value: key for key, value in label_id2cate.items()}
train['label'] = train['categories'].map(label_cate2id)
train = train[['text', 'label', 'paperid']]
train_y = train["label"]
train_df = train[['text', 'label', 'paperid']][:40000]
eval_df = train[['text', 'label', 'paperid']][40000:]
```


```python
print(label_id2cate)
```

    {0: 'cs.CL', 1: 'cs.NE', 2: 'cs.DL', 3: 'cs.CV', 4: 'cs.LG', 5: 'cs.DS', 6: 'cs.IR', 7: 'cs.RO', 8: 'cs.DM', 9: 'cs.CR', 10: 'cs.AR', 11: 'cs.NI', 12: 'cs.AI', 13: 'cs.SE', 14: 'cs.CG', 15: 'cs.LO', 16: 'cs.SY', 17: 'cs.GR', 18: 'cs.PL', 19: 'cs.SI', 20: 'cs.OH', 21: 'cs.HC', 22: 'cs.MA', 23: 'cs.GT', 24: 'cs.ET', 25: 'cs.FL', 26: 'cs.CC', 27: 'cs.DB', 28: 'cs.DC', 29: 'cs.CY', 30: 'cs.CE', 31: 'cs.MM', 32: 'cs.NA', 33: 'cs.PF', 34: 'cs.OS', 35: 'cs.SD', 36: 'cs.SC', 37: 'cs.MS', 38: 'cs.GL'}



```python
print(label_cate2id)
```

    {'cs.CL': 0, 'cs.NE': 1, 'cs.DL': 2, 'cs.CV': 3, 'cs.LG': 4, 'cs.DS': 5, 'cs.IR': 6, 'cs.RO': 7, 'cs.DM': 8, 'cs.CR': 9, 'cs.AR': 10, 'cs.NI': 11, 'cs.AI': 12, 'cs.SE': 13, 'cs.CG': 14, 'cs.LO': 15, 'cs.SY': 16, 'cs.GR': 17, 'cs.PL': 18, 'cs.SI': 19, 'cs.OH': 20, 'cs.HC': 21, 'cs.MA': 22, 'cs.GT': 23, 'cs.ET': 24, 'cs.FL': 25, 'cs.CC': 26, 'cs.DB': 27, 'cs.DC': 28, 'cs.CY': 29, 'cs.CE': 30, 'cs.MM': 31, 'cs.NA': 32, 'cs.PF': 33, 'cs.OS': 34, 'cs.SD': 35, 'cs.SC': 36, 'cs.MS': 37, 'cs.GL': 38}



```python
train_df.describe
```




    <bound method NDFrame.describe of                                                     text  label      paperid
    0      Hard but Robust, Easy but Sensitive: How Encod...      0  train_00000
    1      An Easy-to-use Real-world Multi-objective Opti...      1  train_00001
    2      Exploration of reproducibility issues in scien...      2  train_00002
    3      Scheduled Sampling for Transformers   Schedule...      0  train_00003
    4      Hybrid Forests for Left Ventricle Segmentation...      3  train_00004
    ...                                                  ...    ...          ...
    39995  EyeDoc: Documentation Navigation with Eye Trac...     13  train_39995
    39996  Design of an Ultra-Efficient Reversible Full A...     24  train_39996
    39997  Hybrid FPMS: A New Fairness Protocol Managemen...     11  train_39997
    39998  Conditional Rap Lyrics Generation with Denoisi...      0  train_39998
    39999  Cross-Lingual Syntactic Transfer with Limited ...      0  train_39999
    
    [40000 rows x 3 columns]>




```python
from paddlenlp.datasets import load_dataset

# read train data
def read(pd_data):
    for index, item in pd_data.iterrows():       
        yield {'text': item['text'], 'label': item['label'], 'qid': item['paperid'].strip('train_')}
```

## 5.数据载入


```python
# data_path为read()方法的参数
train_ds = load_dataset(read, pd_data=train_df,lazy=False)
dev_ds = load_dataset(read, pd_data=eval_df,lazy=False)
```


```python
for i in range(5):
    print(train_ds[i])
```

    {'text': 'Hard but Robust, Easy but Sensitive: How Encoder and Decoder Perform in\n  Neural Machine Translation   Neural machine translation (NMT) typically adopts the encoder-decoder\nframework. A good understanding of the characteristics and functionalities of\nthe encoder and decoder can help to explain the pros and cons of the framework,\nand design better models for NMT. In this work, we conduct an empirical study\non the encoder and the decoder in NMT, taking Transformer as an example. We\nfind that 1) the decoder handles an easier task than the encoder in NMT, 2) the\ndecoder is more sensitive to the input noise than the encoder, and 3) the\npreceding words/tokens in the decoder provide strong conditional information,\nwhich accounts for the two observations above. We hope those observations can\nshed light on the characteristics of the encoder and decoder and inspire future\nresearch on NMT.\n', 'label': 0, 'qid': '00000'}
    {'text': 'An Easy-to-use Real-world Multi-objective Optimization Problem Suite   Although synthetic test problems are widely used for the performance\nassessment of evolutionary multi-objective optimization algorithms, they are\nlikely to include unrealistic properties which may lead to\noverestimation/underestimation. To address this issue, we present a\nmulti-objective optimization problem suite consisting of 16 bound-constrained\nreal-world problems. The problem suite includes various problems in terms of\nthe number of objectives, the shape of the Pareto front, and the type of design\nvariables. 4 out of the 16 problems are multi-objective mixed-integer\noptimization problems. We provide Java, C, and Matlab source codes of the 16\nproblems so that they are available in an off-the-shelf manner. We examine an\napproximated Pareto front of each test problem. We also analyze the performance\nof six representative evolutionary multi-objective optimization algorithms on\nthe 16 problems. In addition to the 16 problems, we present 8 constrained\nmulti-objective real-world problems.\n', 'label': 1, 'qid': '00001'}
    {'text': 'Exploration of reproducibility issues in scientometric research Part 1:\n  Direct reproducibility   This is the first part of a small-scale explorative study in an effort to\nstart assessing reproducibility issues specific to scientometrics research.\nThis effort is motivated by the desire to generate empirical data to inform\ndebates about reproducibility in scientometrics. Rather than attempt to\nreproduce studies, we explore how we might assess "in principle"\nreproducibility based on a critical review of the content of published papers.\nThe first part of the study focuses on direct reproducibility - that is the\nability to reproduce the specific evidence produced by an original study using\nthe same data, methods, and procedures. The second part (Velden et al. 2018) is\ndedicated to conceptual reproducibility - that is the robustness of knowledge\nclaims towards verification by an alternative approach using different data,\nmethods and procedures. The study is exploratory: it investigates only a very\nlimited number of publications and serves us to develop instruments for\nidentifying potential reproducibility issues of published studies: These are a\ncategorization of study types and a taxonomy of threats to reproducibility. We\nwork with a select sample of five publications in scientometrics covering a\nvariation of study types of theoretical, methodological, and empirical nature.\nBased on observations made during our exploratory review, we conclude this\npaper with open questions on how to approach and assess the status of direct\nreproducibility in scientometrics, intended for discussion at the special track\non "Reproducibility in Scientometrics" at STI2018 in Leiden.\n', 'label': 2, 'qid': '00002'}
    {'text': 'Scheduled Sampling for Transformers   Scheduled sampling is a technique for avoiding one of the known problems in\nsequence-to-sequence generation: exposure bias. It consists of feeding the\nmodel a mix of the teacher forced embeddings and the model predictions from the\nprevious step in training time. The technique has been used for improving the\nmodel performance with recurrent neural networks (RNN). In the Transformer\nmodel, unlike the RNN, the generation of a new word attends to the full\nsentence generated so far, not only to the last word, and it is not\nstraightforward to apply the scheduled sampling technique. We propose some\nstructural changes to allow scheduled sampling to be applied to Transformer\narchitecture, via a two-pass decoding strategy. Experiments on two language\npairs achieve performance close to a teacher-forcing baseline and show that\nthis technique is promising for further exploration.\n', 'label': 0, 'qid': '00003'}
    {'text': "Hybrid Forests for Left Ventricle Segmentation using only the first\n  slice label   Machine learning models produce state-of-the-art results in many MRI images\nsegmentation. However, most of these models are trained on very large datasets\nwhich come from experts manual labeling. This labeling process is very time\nconsuming and costs experts work. Therefore finding a way to reduce this cost\nis on high demand. In this paper, we propose a segmentation method which\nexploits MRI images sequential structure to nearly drop out this labeling task.\nOnly the first slice needs to be manually labeled to train the model which then\ninfers the next slice's segmentation. Inference result is another datum used to\ntrain the model again. The updated model then infers the third slice and the\nsame process is carried out until the last slice. The proposed model is an\ncombination of two Random Forest algorithms: the classical one and a recent one\nnamely Mondrian Forests. We applied our method on human left ventricle\nsegmentation and results are very promising. This method can also be used to\ngenerate labels.\n", 'label': 3, 'qid': '00004'}


# 三、使用预训练模型

## 1.选取预训练模型


```python
import paddlenlp as ppnlp

from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
# 指定模型名称，一键加载模型
model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en", num_classes=len(train.label.unique()))
# 同样地，通过指定模型名称一键加载对应的Tokenizer，用于处理文本数据，如切分token，转token_id等。
tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en")
```

    [2021-07-25 23:03:23,255] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_2.0_large_en/skep_ernie_2.0_large_en.pdparams
    [2021-07-25 23:03:34,982] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_2.0_large_en/skep_ernie_2.0_large_en.vocab.txt


## 2.数据读取
使用paddle.io.DataLoader接口多线程异步加载数据。


```python
import os
from functools import partial


import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad

from utils import create_dataloader

def convert_example(example,
                    tokenizer,
                    max_seq_length=512,
                    is_test=False):
   
    # 将原数据处理成model可读入的格式，enocded_inputs是一个dict，包含input_ids、token_type_ids等字段
    encoded_inputs = tokenizer(
        text=example["text"], max_seq_len=max_seq_length)

    # input_ids：对文本切分token后，在词汇表中对应的token id
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids：当前token属于句子1还是句子2，即上述图中表达的segment ids
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        # label：情感极性类别
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        # qid：每条数据的编号
        qid = np.array([example["qid"]], dtype="int64")
        return input_ids, token_type_ids, qid
```


```python
# 批量数据大小
batch_size = 10
# 文本序列最大长度
max_seq_length = 450

# 将数据处理成模型可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

# 将数据组成批量式数据，如
# 将不同长度的文本序列padding到批量式数据中最大长度
# 将每条数据label堆叠在一起
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # labels
): [data for data in fn(samples)]
train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
dev_data_loader = create_dataloader(
    dev_ds,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

## 3.设置Fine-Tune优化策略，接入评价指标

# 四、模型训练与评估
模型训练的过程通常有以下步骤：

1. 从dataloader中取出一个batch data
2. 将batch data喂给model，做前向计算
3. 将前向计算结果传给损失函数，计算loss。将前向计算结果传给评价方法，计算评价指标。
4. loss反向回传，更新梯度。重复以上步骤。

每训练一个epoch时，程序将会评估一次，评估当前模型训练的效果。

## 1.参数配置


```python
from paddlenlp.transformers import LinearDecayWithWarmup
import paddle

# 训练轮次
epochs = 3

# len(train_data_loader)一轮训练所需要的step数
num_training_steps = len(train_data_loader) * epochs

# Adam优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=2e-5,
    parameters=model.parameters())
# 交叉熵损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()
# accuracy评价指标
metric = paddle.metric.Accuracy()
```

## 2.加入visualdl


```python
# 加入日志显示
from visualdl import LogWriter

writer = LogWriter("./log")
```

## 3.evaluate方法


```python
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):

    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:      
        input_ids, token_type_ids,  labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.5f, accu: %.5f" % (np.mean(losses), accu))
    # 加入eval日志显示
    writer.add_scalar(tag="eval/loss", step=global_step, value=np.mean(losses))
    writer.add_scalar(tag="eval/acc", step=global_step, value=accu)  
    model.train()
    metric.reset()
    return accu
```

## 4.开始训练


```python

save_dir = "checkpoint"
if not  os.path.exists(save_dir):
    os.makedirs(save_dir)
global_step = 0
pre_accu=0
accu=0
for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        if global_step % 10 == 0 :
            print("global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f" % (global_step, epoch, step, loss, acc))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        # 每间隔 400 step 在验证集和测试集上进行评估
        if global_step % 400 == 0:
            accu=evaluate(model, criterion, metric, dev_data_loader)
            # 加入train日志显示
            writer.add_scalar(tag="train/loss", step=global_step, value=loss)
            writer.add_scalar(tag="train/acc", step=global_step, value=acc)       
        if accu>pre_accu:
            # 加入保存
            save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            paddle.save(model.state_dict(), save_param_path)
            pre_accu=accu
tokenizer.save_pretrained(save_dir)
```

    global step 10, epoch: 1, batch: 10, loss: 3.95217, acc: 0.17000
    global step 20, epoch: 1, batch: 20, loss: 3.04861, acc: 0.19500
    global step 30, epoch: 1, batch: 30, loss: 2.81900, acc: 0.21667
    global step 40, epoch: 1, batch: 40, loss: 2.78886, acc: 0.23750
    global step 50, epoch: 1, batch: 50, loss: 2.60308, acc: 0.27200
    global step 60, epoch: 1, batch: 60, loss: 2.16761, acc: 0.30667
    global step 70, epoch: 1, batch: 70, loss: 1.47025, acc: 0.33857
    global step 80, epoch: 1, batch: 80, loss: 0.97065, acc: 0.36500
    global step 90, epoch: 1, batch: 90, loss: 1.86051, acc: 0.38556
    global step 100, epoch: 1, batch: 100, loss: 1.88056, acc: 0.39600
    global step 110, epoch: 1, batch: 110, loss: 2.12686, acc: 0.40273
    global step 120, epoch: 1, batch: 120, loss: 1.56812, acc: 0.41250
    global step 130, epoch: 1, batch: 130, loss: 2.05380, acc: 0.43077
    global step 140, epoch: 1, batch: 140, loss: 1.71422, acc: 0.44500
    global step 150, epoch: 1, batch: 150, loss: 2.57056, acc: 0.45467
    global step 160, epoch: 1, batch: 160, loss: 1.78537, acc: 0.46375
    global step 170, epoch: 1, batch: 170, loss: 1.21383, acc: 0.47353
    global step 180, epoch: 1, batch: 180, loss: 2.51876, acc: 0.48167
    global step 190, epoch: 1, batch: 190, loss: 1.71976, acc: 0.49053
    global step 200, epoch: 1, batch: 200, loss: 1.08125, acc: 0.49950
    global step 210, epoch: 1, batch: 210, loss: 1.28558, acc: 0.50810
    global step 220, epoch: 1, batch: 220, loss: 1.38988, acc: 0.51091
    global step 230, epoch: 1, batch: 230, loss: 2.26875, acc: 0.51348
    global step 240, epoch: 1, batch: 240, loss: 1.59910, acc: 0.51917
    global step 250, epoch: 1, batch: 250, loss: 2.01791, acc: 0.52280
    global step 260, epoch: 1, batch: 260, loss: 2.05497, acc: 0.52731
    global step 270, epoch: 1, batch: 270, loss: 1.05995, acc: 0.53296
    global step 280, epoch: 1, batch: 280, loss: 1.04271, acc: 0.53750
    global step 290, epoch: 1, batch: 290, loss: 0.30010, acc: 0.54379
    global step 300, epoch: 1, batch: 300, loss: 1.37042, acc: 0.54733
    global step 310, epoch: 1, batch: 310, loss: 2.02820, acc: 0.55226
    global step 320, epoch: 1, batch: 320, loss: 1.74997, acc: 0.55406
    global step 330, epoch: 1, batch: 330, loss: 1.73740, acc: 0.55576
    global step 340, epoch: 1, batch: 340, loss: 1.32048, acc: 0.55941
    global step 350, epoch: 1, batch: 350, loss: 0.58030, acc: 0.56143
    global step 360, epoch: 1, batch: 360, loss: 1.64941, acc: 0.56361
    global step 370, epoch: 1, batch: 370, loss: 2.86609, acc: 0.56541
    global step 380, epoch: 1, batch: 380, loss: 1.73917, acc: 0.56421
    global step 390, epoch: 1, batch: 390, loss: 1.43640, acc: 0.56590
    global step 400, epoch: 1, batch: 400, loss: 1.32176, acc: 0.56750
    eval loss: 1.26403, accu: 0.68100
    global step 410, epoch: 1, batch: 410, loss: 1.16001, acc: 0.67000
    global step 420, epoch: 1, batch: 420, loss: 1.70566, acc: 0.68500
    global step 430, epoch: 1, batch: 430, loss: 1.69782, acc: 0.66000
    global step 440, epoch: 1, batch: 440, loss: 1.43802, acc: 0.68000
    global step 450, epoch: 1, batch: 450, loss: 0.63585, acc: 0.67600
    global step 460, epoch: 1, batch: 460, loss: 0.42743, acc: 0.67333
    global step 470, epoch: 1, batch: 470, loss: 1.77239, acc: 0.66429
    global step 480, epoch: 1, batch: 480, loss: 0.82629, acc: 0.65375
    global step 490, epoch: 1, batch: 490, loss: 1.80944, acc: 0.65667
    global step 500, epoch: 1, batch: 500, loss: 1.73650, acc: 0.65800
    global step 510, epoch: 1, batch: 510, loss: 0.68867, acc: 0.66273
    global step 520, epoch: 1, batch: 520, loss: 1.26257, acc: 0.67250
    global step 530, epoch: 1, batch: 530, loss: 1.61707, acc: 0.67692
    global step 540, epoch: 1, batch: 540, loss: 1.29007, acc: 0.67643
    global step 550, epoch: 1, batch: 550, loss: 1.79194, acc: 0.68067
    global step 560, epoch: 1, batch: 560, loss: 0.74449, acc: 0.67937
    global step 570, epoch: 1, batch: 570, loss: 0.84843, acc: 0.67765
    global step 580, epoch: 1, batch: 580, loss: 1.52876, acc: 0.67889
    global step 590, epoch: 1, batch: 590, loss: 1.24010, acc: 0.68158
    global step 600, epoch: 1, batch: 600, loss: 0.84207, acc: 0.68500
    global step 610, epoch: 1, batch: 610, loss: 1.15624, acc: 0.68714
    global step 620, epoch: 1, batch: 620, loss: 0.89163, acc: 0.68500
    global step 630, epoch: 1, batch: 630, loss: 1.08652, acc: 0.68565
    global step 640, epoch: 1, batch: 640, loss: 0.77536, acc: 0.68958
    global step 650, epoch: 1, batch: 650, loss: 1.41852, acc: 0.68880
    global step 660, epoch: 1, batch: 660, loss: 1.22175, acc: 0.68769
    global step 670, epoch: 1, batch: 670, loss: 1.26150, acc: 0.68630
    global step 680, epoch: 1, batch: 680, loss: 1.50637, acc: 0.68714
    global step 690, epoch: 1, batch: 690, loss: 1.65543, acc: 0.68655
    global step 700, epoch: 1, batch: 700, loss: 0.31176, acc: 0.68700
    global step 710, epoch: 1, batch: 710, loss: 1.90844, acc: 0.68742
    global step 720, epoch: 1, batch: 720, loss: 0.83390, acc: 0.69063
    global step 730, epoch: 1, batch: 730, loss: 1.92620, acc: 0.69000
    global step 740, epoch: 1, batch: 740, loss: 1.29228, acc: 0.68765
    global step 750, epoch: 1, batch: 750, loss: 0.65479, acc: 0.68857
    global step 760, epoch: 1, batch: 760, loss: 1.04747, acc: 0.68833
    global step 770, epoch: 1, batch: 770, loss: 1.10886, acc: 0.68973
    global step 780, epoch: 1, batch: 780, loss: 0.65149, acc: 0.69158
    global step 790, epoch: 1, batch: 790, loss: 1.11695, acc: 0.69231
    global step 800, epoch: 1, batch: 800, loss: 1.18454, acc: 0.69100
    eval loss: 1.10177, accu: 0.70560
    global step 810, epoch: 1, batch: 810, loss: 1.34260, acc: 0.73000
    global step 820, epoch: 1, batch: 820, loss: 0.90944, acc: 0.73000
    global step 830, epoch: 1, batch: 830, loss: 0.66672, acc: 0.73333
    global step 840, epoch: 1, batch: 840, loss: 1.66213, acc: 0.73750
    global step 850, epoch: 1, batch: 850, loss: 1.69877, acc: 0.72800
    global step 860, epoch: 1, batch: 860, loss: 0.77026, acc: 0.72333
    global step 870, epoch: 1, batch: 870, loss: 1.19338, acc: 0.71000
    global step 880, epoch: 1, batch: 880, loss: 0.46703, acc: 0.70375
    global step 890, epoch: 1, batch: 890, loss: 1.65559, acc: 0.70444
    global step 900, epoch: 1, batch: 900, loss: 1.00748, acc: 0.70600
    global step 910, epoch: 1, batch: 910, loss: 0.67676, acc: 0.70818
    global step 920, epoch: 1, batch: 920, loss: 0.76179, acc: 0.71500
    global step 930, epoch: 1, batch: 930, loss: 0.96079, acc: 0.71077
    global step 940, epoch: 1, batch: 940, loss: 1.01108, acc: 0.71357
    global step 950, epoch: 1, batch: 950, loss: 0.96144, acc: 0.71200
    global step 960, epoch: 1, batch: 960, loss: 1.32529, acc: 0.71125
    global step 970, epoch: 1, batch: 970, loss: 0.61233, acc: 0.71353
    global step 980, epoch: 1, batch: 980, loss: 1.37022, acc: 0.70944
    global step 990, epoch: 1, batch: 990, loss: 0.95692, acc: 0.71158
    global step 1000, epoch: 1, batch: 1000, loss: 1.76144, acc: 0.70900
    global step 1010, epoch: 1, batch: 1010, loss: 0.58346, acc: 0.70857
    global step 1020, epoch: 1, batch: 1020, loss: 1.58341, acc: 0.70773
    global step 1030, epoch: 1, batch: 1030, loss: 1.81320, acc: 0.70565
    global step 1040, epoch: 1, batch: 1040, loss: 0.77600, acc: 0.70833
    global step 1050, epoch: 1, batch: 1050, loss: 1.55931, acc: 0.70640
    global step 1060, epoch: 1, batch: 1060, loss: 1.17198, acc: 0.70462
    global step 1070, epoch: 1, batch: 1070, loss: 0.98360, acc: 0.70667
    global step 1080, epoch: 1, batch: 1080, loss: 0.33099, acc: 0.70571
    global step 1090, epoch: 1, batch: 1090, loss: 0.96146, acc: 0.70379
    global step 1100, epoch: 1, batch: 1100, loss: 0.79459, acc: 0.70767
    global step 1110, epoch: 1, batch: 1110, loss: 0.36236, acc: 0.71194
    global step 1120, epoch: 1, batch: 1120, loss: 0.79325, acc: 0.71500
    global step 1130, epoch: 1, batch: 1130, loss: 0.86123, acc: 0.71303
    global step 1140, epoch: 1, batch: 1140, loss: 0.70096, acc: 0.71206
    global step 1150, epoch: 1, batch: 1150, loss: 1.02544, acc: 0.71086
    global step 1160, epoch: 1, batch: 1160, loss: 0.98635, acc: 0.71083
    global step 1170, epoch: 1, batch: 1170, loss: 1.27079, acc: 0.71054
    global step 1180, epoch: 1, batch: 1180, loss: 0.35406, acc: 0.71026
    global step 1190, epoch: 1, batch: 1190, loss: 0.84547, acc: 0.71103
    global step 1200, epoch: 1, batch: 1200, loss: 0.98681, acc: 0.71150
    eval loss: 1.00313, accu: 0.72640
    global step 1210, epoch: 1, batch: 1210, loss: 1.14775, acc: 0.67000
    global step 1220, epoch: 1, batch: 1220, loss: 1.21631, acc: 0.69000
    global step 1230, epoch: 1, batch: 1230, loss: 1.28166, acc: 0.69000
    global step 1240, epoch: 1, batch: 1240, loss: 0.96632, acc: 0.69250
    global step 1250, epoch: 1, batch: 1250, loss: 0.63076, acc: 0.69000
    global step 1260, epoch: 1, batch: 1260, loss: 1.39425, acc: 0.69333
    global step 1270, epoch: 1, batch: 1270, loss: 0.99936, acc: 0.70000
    global step 1280, epoch: 1, batch: 1280, loss: 1.45183, acc: 0.69125
    global step 1290, epoch: 1, batch: 1290, loss: 0.81584, acc: 0.69444
    global step 1300, epoch: 1, batch: 1300, loss: 1.82980, acc: 0.69600
    global step 1310, epoch: 1, batch: 1310, loss: 0.52839, acc: 0.70364
    global step 1320, epoch: 1, batch: 1320, loss: 1.68607, acc: 0.70750
    global step 1330, epoch: 1, batch: 1330, loss: 0.66399, acc: 0.70615
    global step 1340, epoch: 1, batch: 1340, loss: 1.59919, acc: 0.70214
    global step 1350, epoch: 1, batch: 1350, loss: 0.66343, acc: 0.70533
    global step 1360, epoch: 1, batch: 1360, loss: 0.67445, acc: 0.70937
    global step 1370, epoch: 1, batch: 1370, loss: 1.57687, acc: 0.71235
    global step 1380, epoch: 1, batch: 1380, loss: 0.54981, acc: 0.71667
    global step 1390, epoch: 1, batch: 1390, loss: 0.99166, acc: 0.71368
    global step 1400, epoch: 1, batch: 1400, loss: 1.21766, acc: 0.71200
    global step 1410, epoch: 1, batch: 1410, loss: 0.71901, acc: 0.71429
    global step 1420, epoch: 1, batch: 1420, loss: 1.26917, acc: 0.71636
    global step 1430, epoch: 1, batch: 1430, loss: 1.01909, acc: 0.71739
    global step 1440, epoch: 1, batch: 1440, loss: 0.69920, acc: 0.71708
    global step 1450, epoch: 1, batch: 1450, loss: 1.14193, acc: 0.71560
    global step 1460, epoch: 1, batch: 1460, loss: 0.54562, acc: 0.71654
    global step 1470, epoch: 1, batch: 1470, loss: 0.31759, acc: 0.71556
    global step 1480, epoch: 1, batch: 1480, loss: 1.60676, acc: 0.71786
    global step 1490, epoch: 1, batch: 1490, loss: 0.67706, acc: 0.71931
    global step 1500, epoch: 1, batch: 1500, loss: 0.65905, acc: 0.72100
    global step 1510, epoch: 1, batch: 1510, loss: 0.94079, acc: 0.72290
    global step 1520, epoch: 1, batch: 1520, loss: 0.33539, acc: 0.72313
    global step 1530, epoch: 1, batch: 1530, loss: 0.47959, acc: 0.72273
    global step 1540, epoch: 1, batch: 1540, loss: 0.66254, acc: 0.72059
    global step 1550, epoch: 1, batch: 1550, loss: 0.46359, acc: 0.72143
    global step 1560, epoch: 1, batch: 1560, loss: 0.53611, acc: 0.72083
    global step 1570, epoch: 1, batch: 1570, loss: 1.90791, acc: 0.71946
    global step 1580, epoch: 1, batch: 1580, loss: 2.76214, acc: 0.71921
    global step 1590, epoch: 1, batch: 1590, loss: 0.88339, acc: 0.71897
    global step 1600, epoch: 1, batch: 1600, loss: 0.60178, acc: 0.72050
    eval loss: 0.93497, accu: 0.74150
    global step 1610, epoch: 1, batch: 1610, loss: 0.68509, acc: 0.64000
    global step 1620, epoch: 1, batch: 1620, loss: 0.94018, acc: 0.69000
    global step 1630, epoch: 1, batch: 1630, loss: 0.64191, acc: 0.72000
    global step 1640, epoch: 1, batch: 1640, loss: 0.90913, acc: 0.72750
    global step 1650, epoch: 1, batch: 1650, loss: 1.14658, acc: 0.73800
    global step 1660, epoch: 1, batch: 1660, loss: 0.87995, acc: 0.71667
    global step 1670, epoch: 1, batch: 1670, loss: 0.89111, acc: 0.72571
    global step 1680, epoch: 1, batch: 1680, loss: 1.50038, acc: 0.70875
    global step 1690, epoch: 1, batch: 1690, loss: 0.19620, acc: 0.70667
    global step 1700, epoch: 1, batch: 1700, loss: 1.11558, acc: 0.70700
    global step 1710, epoch: 1, batch: 1710, loss: 0.45125, acc: 0.70727
    global step 1720, epoch: 1, batch: 1720, loss: 1.00658, acc: 0.71417
    global step 1730, epoch: 1, batch: 1730, loss: 1.08214, acc: 0.70923
    global step 1740, epoch: 1, batch: 1740, loss: 1.06100, acc: 0.71357
    global step 1750, epoch: 1, batch: 1750, loss: 0.89252, acc: 0.72200
    global step 1760, epoch: 1, batch: 1760, loss: 1.82281, acc: 0.72437
    global step 1770, epoch: 1, batch: 1770, loss: 1.44877, acc: 0.72412
    global step 1780, epoch: 1, batch: 1780, loss: 0.93772, acc: 0.72611
    global step 1790, epoch: 1, batch: 1790, loss: 0.29955, acc: 0.72789
    global step 1800, epoch: 1, batch: 1800, loss: 0.98046, acc: 0.72700
    global step 1810, epoch: 1, batch: 1810, loss: 1.14146, acc: 0.72714
    global step 1820, epoch: 1, batch: 1820, loss: 0.67932, acc: 0.73000
    global step 1830, epoch: 1, batch: 1830, loss: 0.97355, acc: 0.72957
    global step 1840, epoch: 1, batch: 1840, loss: 0.76277, acc: 0.73000
    global step 1850, epoch: 1, batch: 1850, loss: 1.40421, acc: 0.72920
    global step 1860, epoch: 1, batch: 1860, loss: 0.67980, acc: 0.72846
    global step 1870, epoch: 1, batch: 1870, loss: 0.32650, acc: 0.72815
    global step 1880, epoch: 1, batch: 1880, loss: 0.29382, acc: 0.73036
    global step 1890, epoch: 1, batch: 1890, loss: 1.28295, acc: 0.73000
    global step 1900, epoch: 1, batch: 1900, loss: 0.88361, acc: 0.72933
    global step 1910, epoch: 1, batch: 1910, loss: 0.76948, acc: 0.72581
    global step 1920, epoch: 1, batch: 1920, loss: 0.25510, acc: 0.72562
    global step 1930, epoch: 1, batch: 1930, loss: 1.00567, acc: 0.72424
    global step 1940, epoch: 1, batch: 1940, loss: 1.26477, acc: 0.72471
    global step 1950, epoch: 1, batch: 1950, loss: 1.22270, acc: 0.72371
    global step 1960, epoch: 1, batch: 1960, loss: 0.91618, acc: 0.72361
    global step 1970, epoch: 1, batch: 1970, loss: 1.20559, acc: 0.72432
    global step 1980, epoch: 1, batch: 1980, loss: 0.90101, acc: 0.72342
    global step 1990, epoch: 1, batch: 1990, loss: 0.82997, acc: 0.72359
    global step 2000, epoch: 1, batch: 2000, loss: 1.58427, acc: 0.72375
    eval loss: 0.85893, accu: 0.75670
    global step 2010, epoch: 1, batch: 2010, loss: 2.41633, acc: 0.77000
    global step 2020, epoch: 1, batch: 2020, loss: 0.88530, acc: 0.76000
    global step 2030, epoch: 1, batch: 2030, loss: 0.54826, acc: 0.74000
    global step 2040, epoch: 1, batch: 2040, loss: 2.43883, acc: 0.73750
    global step 2050, epoch: 1, batch: 2050, loss: 0.84271, acc: 0.74000
    global step 2060, epoch: 1, batch: 2060, loss: 0.70129, acc: 0.74500
    global step 2070, epoch: 1, batch: 2070, loss: 0.86887, acc: 0.74857
    global step 2080, epoch: 1, batch: 2080, loss: 0.79423, acc: 0.75625
    global step 2090, epoch: 1, batch: 2090, loss: 1.28482, acc: 0.75333
    global step 2100, epoch: 1, batch: 2100, loss: 0.72225, acc: 0.75800
    global step 2110, epoch: 1, batch: 2110, loss: 0.24254, acc: 0.75364
    global step 2120, epoch: 1, batch: 2120, loss: 0.90524, acc: 0.75250
    global step 2130, epoch: 1, batch: 2130, loss: 1.16132, acc: 0.75154
    global step 2140, epoch: 1, batch: 2140, loss: 0.63820, acc: 0.75286
    global step 2150, epoch: 1, batch: 2150, loss: 0.30088, acc: 0.75000
    global step 2160, epoch: 1, batch: 2160, loss: 1.14014, acc: 0.74625
    global step 2170, epoch: 1, batch: 2170, loss: 0.38857, acc: 0.75000
    global step 2180, epoch: 1, batch: 2180, loss: 1.00054, acc: 0.75056
    global step 2190, epoch: 1, batch: 2190, loss: 1.79212, acc: 0.74842
    global step 2200, epoch: 1, batch: 2200, loss: 0.46752, acc: 0.75050
    global step 2210, epoch: 1, batch: 2210, loss: 0.67916, acc: 0.75048
    global step 2220, epoch: 1, batch: 2220, loss: 0.31962, acc: 0.75000
    global step 2230, epoch: 1, batch: 2230, loss: 0.70790, acc: 0.75174
    global step 2240, epoch: 1, batch: 2240, loss: 0.72144, acc: 0.74875
    global step 2250, epoch: 1, batch: 2250, loss: 0.44476, acc: 0.74960
    global step 2260, epoch: 1, batch: 2260, loss: 0.61596, acc: 0.74731
    global step 2270, epoch: 1, batch: 2270, loss: 0.65794, acc: 0.74778
    global step 2280, epoch: 1, batch: 2280, loss: 0.67166, acc: 0.74714
    global step 2290, epoch: 1, batch: 2290, loss: 0.71670, acc: 0.75000
    global step 2300, epoch: 1, batch: 2300, loss: 1.00967, acc: 0.75133
    global step 2310, epoch: 1, batch: 2310, loss: 1.04643, acc: 0.75129
    global step 2320, epoch: 1, batch: 2320, loss: 1.69749, acc: 0.75031
    global step 2330, epoch: 1, batch: 2330, loss: 1.24835, acc: 0.74909
    global step 2340, epoch: 1, batch: 2340, loss: 1.00815, acc: 0.74912
    global step 2350, epoch: 1, batch: 2350, loss: 1.13155, acc: 0.74714
    global step 2360, epoch: 1, batch: 2360, loss: 0.37222, acc: 0.74806
    global step 2370, epoch: 1, batch: 2370, loss: 0.99892, acc: 0.74730
    global step 2380, epoch: 1, batch: 2380, loss: 0.61200, acc: 0.74684
    global step 2390, epoch: 1, batch: 2390, loss: 1.15501, acc: 0.74641
    global step 2400, epoch: 1, batch: 2400, loss: 0.62091, acc: 0.74725
    eval loss: 0.83156, accu: 0.76100
    global step 2410, epoch: 1, batch: 2410, loss: 0.57750, acc: 0.78000
    global step 2420, epoch: 1, batch: 2420, loss: 0.47890, acc: 0.75000
    global step 2430, epoch: 1, batch: 2430, loss: 1.69563, acc: 0.74667
    global step 2440, epoch: 1, batch: 2440, loss: 0.79738, acc: 0.75750
    global step 2450, epoch: 1, batch: 2450, loss: 1.09757, acc: 0.75600
    global step 2460, epoch: 1, batch: 2460, loss: 1.38112, acc: 0.75833
    global step 2470, epoch: 1, batch: 2470, loss: 0.63452, acc: 0.76429
    global step 2480, epoch: 1, batch: 2480, loss: 0.27620, acc: 0.77250
    global step 2490, epoch: 1, batch: 2490, loss: 0.11567, acc: 0.77667
    global step 2500, epoch: 1, batch: 2500, loss: 0.79962, acc: 0.77900
    global step 2510, epoch: 1, batch: 2510, loss: 0.71080, acc: 0.77000
    global step 2520, epoch: 1, batch: 2520, loss: 0.83227, acc: 0.76833
    global step 2530, epoch: 1, batch: 2530, loss: 0.69653, acc: 0.76692
    global step 2540, epoch: 1, batch: 2540, loss: 0.27448, acc: 0.76571
    global step 2550, epoch: 1, batch: 2550, loss: 0.79022, acc: 0.76467
    global step 2560, epoch: 1, batch: 2560, loss: 1.10383, acc: 0.76375
    global step 2570, epoch: 1, batch: 2570, loss: 1.19593, acc: 0.76118
    global step 2580, epoch: 1, batch: 2580, loss: 0.31386, acc: 0.76222
    global step 2590, epoch: 1, batch: 2590, loss: 1.35700, acc: 0.76105
    global step 2600, epoch: 1, batch: 2600, loss: 1.12130, acc: 0.76050
    global step 2610, epoch: 1, batch: 2610, loss: 0.31887, acc: 0.76333
    global step 2620, epoch: 1, batch: 2620, loss: 1.31611, acc: 0.76182
    global step 2630, epoch: 1, batch: 2630, loss: 0.99608, acc: 0.75913
    global step 2640, epoch: 1, batch: 2640, loss: 0.47250, acc: 0.76167
    global step 2650, epoch: 1, batch: 2650, loss: 1.54822, acc: 0.75960
    global step 2660, epoch: 1, batch: 2660, loss: 0.15348, acc: 0.76231
    global step 2670, epoch: 1, batch: 2670, loss: 0.40546, acc: 0.76259
    global step 2680, epoch: 1, batch: 2680, loss: 1.14897, acc: 0.76286
    global step 2690, epoch: 1, batch: 2690, loss: 0.87311, acc: 0.76172
    global step 2700, epoch: 1, batch: 2700, loss: 0.44190, acc: 0.76000
    global step 2710, epoch: 1, batch: 2710, loss: 0.50518, acc: 0.76065
    global step 2720, epoch: 1, batch: 2720, loss: 0.54439, acc: 0.76187
    global step 2730, epoch: 1, batch: 2730, loss: 1.11054, acc: 0.75970
    global step 2740, epoch: 1, batch: 2740, loss: 1.33000, acc: 0.75794
    global step 2750, epoch: 1, batch: 2750, loss: 1.56865, acc: 0.75714
    global step 2760, epoch: 1, batch: 2760, loss: 0.50428, acc: 0.75556
    global step 2770, epoch: 1, batch: 2770, loss: 0.64712, acc: 0.75514
    global step 2780, epoch: 1, batch: 2780, loss: 0.98913, acc: 0.75447
    global step 2790, epoch: 1, batch: 2790, loss: 1.34710, acc: 0.75256
    global step 2800, epoch: 1, batch: 2800, loss: 0.79865, acc: 0.75300
    eval loss: 0.80218, accu: 0.76850
    global step 2810, epoch: 1, batch: 2810, loss: 0.75881, acc: 0.69000
    global step 2820, epoch: 1, batch: 2820, loss: 1.75032, acc: 0.69500
    global step 2830, epoch: 1, batch: 2830, loss: 0.91629, acc: 0.71667
    global step 2840, epoch: 1, batch: 2840, loss: 0.58243, acc: 0.73500
    global step 2850, epoch: 1, batch: 2850, loss: 1.18846, acc: 0.73400
    global step 2860, epoch: 1, batch: 2860, loss: 1.75465, acc: 0.73667
    global step 2870, epoch: 1, batch: 2870, loss: 1.03575, acc: 0.73429
    global step 2880, epoch: 1, batch: 2880, loss: 0.70005, acc: 0.73375
    global step 2890, epoch: 1, batch: 2890, loss: 0.34780, acc: 0.74111
    global step 2900, epoch: 1, batch: 2900, loss: 0.82374, acc: 0.74000
    global step 2910, epoch: 1, batch: 2910, loss: 0.20291, acc: 0.73545
    global step 2920, epoch: 1, batch: 2920, loss: 0.76053, acc: 0.73500
    global step 2930, epoch: 1, batch: 2930, loss: 0.72015, acc: 0.74154
    global step 2940, epoch: 1, batch: 2940, loss: 0.73626, acc: 0.74000
    global step 2950, epoch: 1, batch: 2950, loss: 1.47396, acc: 0.74333
    global step 2960, epoch: 1, batch: 2960, loss: 0.28599, acc: 0.74625
    global step 2970, epoch: 1, batch: 2970, loss: 0.57870, acc: 0.74882
    global step 2980, epoch: 1, batch: 2980, loss: 1.22646, acc: 0.74944
    global step 2990, epoch: 1, batch: 2990, loss: 0.80976, acc: 0.75053
    global step 3000, epoch: 1, batch: 3000, loss: 1.58883, acc: 0.74600
    global step 3010, epoch: 1, batch: 3010, loss: 0.11303, acc: 0.74905
    global step 3020, epoch: 1, batch: 3020, loss: 0.89400, acc: 0.74818
    global step 3030, epoch: 1, batch: 3030, loss: 1.75061, acc: 0.74696
    global step 3040, epoch: 1, batch: 3040, loss: 1.35345, acc: 0.74542
    global step 3050, epoch: 1, batch: 3050, loss: 0.88910, acc: 0.74400
    global step 3060, epoch: 1, batch: 3060, loss: 1.09469, acc: 0.74154
    global step 3070, epoch: 1, batch: 3070, loss: 0.46078, acc: 0.74111
    global step 3080, epoch: 1, batch: 3080, loss: 0.53525, acc: 0.74214
    global step 3090, epoch: 1, batch: 3090, loss: 0.87788, acc: 0.74138
    global step 3100, epoch: 1, batch: 3100, loss: 0.09999, acc: 0.73967
    global step 3110, epoch: 1, batch: 3110, loss: 1.03270, acc: 0.73839
    global step 3120, epoch: 1, batch: 3120, loss: 1.45790, acc: 0.74062
    global step 3130, epoch: 1, batch: 3130, loss: 0.60786, acc: 0.74121
    global step 3140, epoch: 1, batch: 3140, loss: 1.39332, acc: 0.74176
    global step 3150, epoch: 1, batch: 3150, loss: 0.96089, acc: 0.74371
    global step 3160, epoch: 1, batch: 3160, loss: 1.04292, acc: 0.74389
    global step 3170, epoch: 1, batch: 3170, loss: 0.66286, acc: 0.74405
    global step 3180, epoch: 1, batch: 3180, loss: 0.58874, acc: 0.74447
    global step 3190, epoch: 1, batch: 3190, loss: 1.64375, acc: 0.74641
    global step 3200, epoch: 1, batch: 3200, loss: 1.31891, acc: 0.74600
    eval loss: 0.80358, accu: 0.76780
    global step 3210, epoch: 1, batch: 3210, loss: 1.35368, acc: 0.74000
    global step 3220, epoch: 1, batch: 3220, loss: 0.32008, acc: 0.78500
    global step 3230, epoch: 1, batch: 3230, loss: 0.50842, acc: 0.76333
    global step 3240, epoch: 1, batch: 3240, loss: 1.48104, acc: 0.75500
    global step 3250, epoch: 1, batch: 3250, loss: 1.16875, acc: 0.75600
    global step 3260, epoch: 1, batch: 3260, loss: 0.78010, acc: 0.76000
    global step 3270, epoch: 1, batch: 3270, loss: 0.22118, acc: 0.75714
    global step 3280, epoch: 1, batch: 3280, loss: 1.03868, acc: 0.76125
    global step 3290, epoch: 1, batch: 3290, loss: 1.15826, acc: 0.75222
    global step 3300, epoch: 1, batch: 3300, loss: 1.34988, acc: 0.75400
    global step 3310, epoch: 1, batch: 3310, loss: 1.02239, acc: 0.75273
    global step 3320, epoch: 1, batch: 3320, loss: 0.42354, acc: 0.75333
    global step 3330, epoch: 1, batch: 3330, loss: 0.32387, acc: 0.75308
    global step 3340, epoch: 1, batch: 3340, loss: 1.45110, acc: 0.75000
    global step 3350, epoch: 1, batch: 3350, loss: 0.64362, acc: 0.74933
    global step 3360, epoch: 1, batch: 3360, loss: 0.58721, acc: 0.75313
    global step 3370, epoch: 1, batch: 3370, loss: 0.52069, acc: 0.75353
    global step 3380, epoch: 1, batch: 3380, loss: 0.42510, acc: 0.75333
    global step 3390, epoch: 1, batch: 3390, loss: 1.20905, acc: 0.74947
    global step 3400, epoch: 1, batch: 3400, loss: 1.29254, acc: 0.75350
    global step 3410, epoch: 1, batch: 3410, loss: 1.03362, acc: 0.75286
    global step 3420, epoch: 1, batch: 3420, loss: 0.69136, acc: 0.75318
    global step 3430, epoch: 1, batch: 3430, loss: 0.38052, acc: 0.75043
    global step 3440, epoch: 1, batch: 3440, loss: 1.06438, acc: 0.75125
    global step 3450, epoch: 1, batch: 3450, loss: 0.36598, acc: 0.75240
    global step 3460, epoch: 1, batch: 3460, loss: 1.03379, acc: 0.75269
    global step 3470, epoch: 1, batch: 3470, loss: 1.01978, acc: 0.75296
    global step 3480, epoch: 1, batch: 3480, loss: 0.54946, acc: 0.75321
    global step 3490, epoch: 1, batch: 3490, loss: 1.50779, acc: 0.75276
    global step 3500, epoch: 1, batch: 3500, loss: 0.78978, acc: 0.75467
    global step 3510, epoch: 1, batch: 3510, loss: 1.53833, acc: 0.75484
    global step 3520, epoch: 1, batch: 3520, loss: 0.24101, acc: 0.75344
    global step 3530, epoch: 1, batch: 3530, loss: 1.04641, acc: 0.75303
    global step 3540, epoch: 1, batch: 3540, loss: 0.52956, acc: 0.75265
    global step 3550, epoch: 1, batch: 3550, loss: 0.73666, acc: 0.75400
    global step 3560, epoch: 1, batch: 3560, loss: 1.22658, acc: 0.75167
    global step 3570, epoch: 1, batch: 3570, loss: 1.11109, acc: 0.75270
    global step 3580, epoch: 1, batch: 3580, loss: 1.26745, acc: 0.75368
    global step 3590, epoch: 1, batch: 3590, loss: 0.80185, acc: 0.75410
    global step 3600, epoch: 1, batch: 3600, loss: 1.98256, acc: 0.75300
    eval loss: 0.78596, accu: 0.76600
    global step 3610, epoch: 1, batch: 3610, loss: 0.85531, acc: 0.82000
    global step 3620, epoch: 1, batch: 3620, loss: 0.46719, acc: 0.80500
    global step 3630, epoch: 1, batch: 3630, loss: 0.38205, acc: 0.80000
    global step 3640, epoch: 1, batch: 3640, loss: 0.38818, acc: 0.78500
    global step 3650, epoch: 1, batch: 3650, loss: 1.46406, acc: 0.77000
    global step 3660, epoch: 1, batch: 3660, loss: 1.08401, acc: 0.76167
    global step 3670, epoch: 1, batch: 3670, loss: 1.05120, acc: 0.76429
    global step 3680, epoch: 1, batch: 3680, loss: 1.04369, acc: 0.76250
    global step 3690, epoch: 1, batch: 3690, loss: 1.42910, acc: 0.75333
    global step 3700, epoch: 1, batch: 3700, loss: 0.72049, acc: 0.75000
    global step 3710, epoch: 1, batch: 3710, loss: 1.46288, acc: 0.74909
    global step 3720, epoch: 1, batch: 3720, loss: 0.32802, acc: 0.74833
    global step 3730, epoch: 1, batch: 3730, loss: 0.65357, acc: 0.74769
    global step 3740, epoch: 1, batch: 3740, loss: 0.76440, acc: 0.74857
    global step 3750, epoch: 1, batch: 3750, loss: 0.44444, acc: 0.75067
    global step 3760, epoch: 1, batch: 3760, loss: 0.58692, acc: 0.75187
    global step 3770, epoch: 1, batch: 3770, loss: 0.73399, acc: 0.75353
    global step 3780, epoch: 1, batch: 3780, loss: 0.70274, acc: 0.75389
    global step 3790, epoch: 1, batch: 3790, loss: 1.05627, acc: 0.75105
    global step 3800, epoch: 1, batch: 3800, loss: 1.06810, acc: 0.75050
    global step 3810, epoch: 1, batch: 3810, loss: 0.60778, acc: 0.75238
    global step 3820, epoch: 1, batch: 3820, loss: 0.67322, acc: 0.75364
    global step 3830, epoch: 1, batch: 3830, loss: 0.53378, acc: 0.75304
    global step 3840, epoch: 1, batch: 3840, loss: 1.12706, acc: 0.75167
    global step 3850, epoch: 1, batch: 3850, loss: 1.19043, acc: 0.75120
    global step 3860, epoch: 1, batch: 3860, loss: 0.49091, acc: 0.75192
    global step 3870, epoch: 1, batch: 3870, loss: 1.00330, acc: 0.75444
    global step 3880, epoch: 1, batch: 3880, loss: 0.90252, acc: 0.75464
    global step 3890, epoch: 1, batch: 3890, loss: 1.13216, acc: 0.75690
    global step 3900, epoch: 1, batch: 3900, loss: 1.19633, acc: 0.75700
    global step 3910, epoch: 1, batch: 3910, loss: 0.84193, acc: 0.75903
    global step 3920, epoch: 1, batch: 3920, loss: 0.05061, acc: 0.76125
    global step 3930, epoch: 1, batch: 3930, loss: 0.80577, acc: 0.76091
    global step 3940, epoch: 1, batch: 3940, loss: 0.22614, acc: 0.76059
    global step 3950, epoch: 1, batch: 3950, loss: 0.66110, acc: 0.75971
    global step 3960, epoch: 1, batch: 3960, loss: 0.97538, acc: 0.75889
    global step 3970, epoch: 1, batch: 3970, loss: 0.62583, acc: 0.75892
    global step 3980, epoch: 1, batch: 3980, loss: 1.31439, acc: 0.75974
    global step 3990, epoch: 1, batch: 3990, loss: 0.72468, acc: 0.76026
    global step 4000, epoch: 1, batch: 4000, loss: 0.93434, acc: 0.76025
    eval loss: 0.79263, accu: 0.76380
    global step 4010, epoch: 2, batch: 10, loss: 0.34645, acc: 0.86000
    global step 4020, epoch: 2, batch: 20, loss: 0.79548, acc: 0.83000
    global step 4030, epoch: 2, batch: 30, loss: 1.44543, acc: 0.80667
    global step 4040, epoch: 2, batch: 40, loss: 0.68586, acc: 0.81250
    global step 4050, epoch: 2, batch: 50, loss: 0.07340, acc: 0.82400
    global step 4060, epoch: 2, batch: 60, loss: 1.14322, acc: 0.82833
    global step 4070, epoch: 2, batch: 70, loss: 0.76962, acc: 0.83000
    global step 4080, epoch: 2, batch: 80, loss: 0.81950, acc: 0.83375
    global step 4090, epoch: 2, batch: 90, loss: 0.52719, acc: 0.83889
    global step 4100, epoch: 2, batch: 100, loss: 0.07916, acc: 0.84600
    global step 4110, epoch: 2, batch: 110, loss: 0.50166, acc: 0.84727
    global step 4120, epoch: 2, batch: 120, loss: 1.18137, acc: 0.84417
    global step 4130, epoch: 2, batch: 130, loss: 0.78622, acc: 0.83923
    global step 4140, epoch: 2, batch: 140, loss: 0.68221, acc: 0.83643
    global step 4150, epoch: 2, batch: 150, loss: 0.32463, acc: 0.83733
    global step 4160, epoch: 2, batch: 160, loss: 0.32681, acc: 0.83313
    global step 4170, epoch: 2, batch: 170, loss: 0.32052, acc: 0.83471
    global step 4180, epoch: 2, batch: 180, loss: 0.54052, acc: 0.83333
    global step 4190, epoch: 2, batch: 190, loss: 0.60498, acc: 0.83368
    global step 4200, epoch: 2, batch: 200, loss: 0.77371, acc: 0.83050
    global step 4210, epoch: 2, batch: 210, loss: 0.67098, acc: 0.82952
    global step 4220, epoch: 2, batch: 220, loss: 1.44546, acc: 0.82727
    global step 4230, epoch: 2, batch: 230, loss: 0.47951, acc: 0.82609
    global step 4240, epoch: 2, batch: 240, loss: 0.94571, acc: 0.82458
    global step 4250, epoch: 2, batch: 250, loss: 0.10608, acc: 0.82440
    global step 4260, epoch: 2, batch: 260, loss: 0.22583, acc: 0.82577
    global step 4270, epoch: 2, batch: 270, loss: 0.75122, acc: 0.82519
    global step 4280, epoch: 2, batch: 280, loss: 0.52014, acc: 0.82429
    global step 4290, epoch: 2, batch: 290, loss: 1.17569, acc: 0.82276
    global step 4300, epoch: 2, batch: 300, loss: 0.62373, acc: 0.82433
    global step 4310, epoch: 2, batch: 310, loss: 0.49491, acc: 0.82419
    global step 4320, epoch: 2, batch: 320, loss: 0.89161, acc: 0.82250
    global step 4330, epoch: 2, batch: 330, loss: 1.66009, acc: 0.81939
    global step 4340, epoch: 2, batch: 340, loss: 0.11755, acc: 0.82000
    global step 4350, epoch: 2, batch: 350, loss: 0.32219, acc: 0.82086
    global step 4360, epoch: 2, batch: 360, loss: 1.51830, acc: 0.82000
    global step 4370, epoch: 2, batch: 370, loss: 0.76925, acc: 0.82081
    global step 4380, epoch: 2, batch: 380, loss: 0.98078, acc: 0.82158
    global step 4390, epoch: 2, batch: 390, loss: 0.13040, acc: 0.82231
    global step 4400, epoch: 2, batch: 400, loss: 0.21304, acc: 0.82150
    eval loss: 0.85869, accu: 0.76170
    global step 4410, epoch: 2, batch: 410, loss: 0.86023, acc: 0.82000
    global step 4420, epoch: 2, batch: 420, loss: 0.60580, acc: 0.80500
    global step 4430, epoch: 2, batch: 430, loss: 1.58568, acc: 0.78333
    global step 4440, epoch: 2, batch: 440, loss: 0.77300, acc: 0.77750
    global step 4450, epoch: 2, batch: 450, loss: 1.04976, acc: 0.78600
    global step 4460, epoch: 2, batch: 460, loss: 0.61302, acc: 0.77333
    global step 4470, epoch: 2, batch: 470, loss: 1.43772, acc: 0.77286
    global step 4480, epoch: 2, batch: 480, loss: 1.89350, acc: 0.76875
    global step 4490, epoch: 2, batch: 490, loss: 0.45449, acc: 0.78111
    global step 4500, epoch: 2, batch: 500, loss: 0.81310, acc: 0.77500
    global step 4510, epoch: 2, batch: 510, loss: 0.17686, acc: 0.77273
    global step 4520, epoch: 2, batch: 520, loss: 1.85175, acc: 0.76917
    global step 4530, epoch: 2, batch: 530, loss: 0.86824, acc: 0.77154
    global step 4540, epoch: 2, batch: 540, loss: 1.13646, acc: 0.77429
    global step 4550, epoch: 2, batch: 550, loss: 0.29015, acc: 0.77467
    global step 4560, epoch: 2, batch: 560, loss: 0.38199, acc: 0.78063
    global step 4570, epoch: 2, batch: 570, loss: 0.61725, acc: 0.78176
    global step 4580, epoch: 2, batch: 580, loss: 0.70710, acc: 0.78111
    global step 4590, epoch: 2, batch: 590, loss: 0.12613, acc: 0.78316
    global step 4600, epoch: 2, batch: 600, loss: 1.17278, acc: 0.78500
    global step 4610, epoch: 2, batch: 610, loss: 0.46448, acc: 0.79000
    global step 4620, epoch: 2, batch: 620, loss: 0.77613, acc: 0.78909
    global step 4630, epoch: 2, batch: 630, loss: 0.43804, acc: 0.79217
    global step 4640, epoch: 2, batch: 640, loss: 0.26173, acc: 0.79125
    global step 4650, epoch: 2, batch: 650, loss: 0.59192, acc: 0.79200
    global step 4660, epoch: 2, batch: 660, loss: 0.54165, acc: 0.79308
    global step 4670, epoch: 2, batch: 670, loss: 0.45078, acc: 0.79370
    global step 4680, epoch: 2, batch: 680, loss: 0.15851, acc: 0.79500
    global step 4690, epoch: 2, batch: 690, loss: 0.65650, acc: 0.79310
    global step 4700, epoch: 2, batch: 700, loss: 0.91858, acc: 0.79167
    global step 4710, epoch: 2, batch: 710, loss: 0.44728, acc: 0.79452
    global step 4720, epoch: 2, batch: 720, loss: 0.75396, acc: 0.79375
    global step 4730, epoch: 2, batch: 730, loss: 1.03102, acc: 0.79424
    global step 4740, epoch: 2, batch: 740, loss: 0.20651, acc: 0.79441
    global step 4750, epoch: 2, batch: 750, loss: 0.21607, acc: 0.79686
    global step 4760, epoch: 2, batch: 760, loss: 0.70606, acc: 0.79583
    global step 4770, epoch: 2, batch: 770, loss: 0.54407, acc: 0.79541
    global step 4780, epoch: 2, batch: 780, loss: 0.47449, acc: 0.79632
    global step 4790, epoch: 2, batch: 790, loss: 0.28733, acc: 0.79667
    global step 4800, epoch: 2, batch: 800, loss: 0.78843, acc: 0.79625
    eval loss: 0.83746, accu: 0.76250
    global step 4810, epoch: 2, batch: 810, loss: 1.03357, acc: 0.85000
    global step 4820, epoch: 2, batch: 820, loss: 0.90538, acc: 0.81500
    global step 4830, epoch: 2, batch: 830, loss: 0.68362, acc: 0.83000
    global step 4840, epoch: 2, batch: 840, loss: 0.70409, acc: 0.82500
    global step 4850, epoch: 2, batch: 850, loss: 0.12529, acc: 0.82200
    global step 4860, epoch: 2, batch: 860, loss: 0.19627, acc: 0.81167
    global step 4870, epoch: 2, batch: 870, loss: 0.93371, acc: 0.81429
    global step 4880, epoch: 2, batch: 880, loss: 0.41981, acc: 0.82375
    global step 4890, epoch: 2, batch: 890, loss: 0.96763, acc: 0.82333
    global step 4900, epoch: 2, batch: 900, loss: 0.66758, acc: 0.82700
    global step 4910, epoch: 2, batch: 910, loss: 0.12026, acc: 0.82909
    global step 4920, epoch: 2, batch: 920, loss: 0.96291, acc: 0.82750
    global step 4930, epoch: 2, batch: 930, loss: 0.85685, acc: 0.82538
    global step 4940, epoch: 2, batch: 940, loss: 0.73748, acc: 0.82286
    global step 4950, epoch: 2, batch: 950, loss: 0.41430, acc: 0.82200
    global step 4960, epoch: 2, batch: 960, loss: 0.45501, acc: 0.82500
    global step 4970, epoch: 2, batch: 970, loss: 0.50541, acc: 0.82412
    global step 4980, epoch: 2, batch: 980, loss: 0.60542, acc: 0.82444
    global step 4990, epoch: 2, batch: 990, loss: 1.04394, acc: 0.82684
    global step 5000, epoch: 2, batch: 1000, loss: 0.69281, acc: 0.83150
    global step 5010, epoch: 2, batch: 1010, loss: 0.42492, acc: 0.83381
    global step 5020, epoch: 2, batch: 1020, loss: 0.34418, acc: 0.83227
    global step 5030, epoch: 2, batch: 1030, loss: 0.90766, acc: 0.83348
    global step 5040, epoch: 2, batch: 1040, loss: 0.27386, acc: 0.83500
    global step 5050, epoch: 2, batch: 1050, loss: 0.58640, acc: 0.83600
    global step 5060, epoch: 2, batch: 1060, loss: 0.33283, acc: 0.83346
    global step 5070, epoch: 2, batch: 1070, loss: 0.37031, acc: 0.83185
    global step 5080, epoch: 2, batch: 1080, loss: 0.89896, acc: 0.83214
    global step 5090, epoch: 2, batch: 1090, loss: 0.61131, acc: 0.83172
    global step 5100, epoch: 2, batch: 1100, loss: 0.49638, acc: 0.83067
    global step 5110, epoch: 2, batch: 1110, loss: 0.52081, acc: 0.83129
    global step 5120, epoch: 2, batch: 1120, loss: 0.68646, acc: 0.83281
    global step 5130, epoch: 2, batch: 1130, loss: 0.70304, acc: 0.83424
    global step 5140, epoch: 2, batch: 1140, loss: 1.03063, acc: 0.83353
    global step 5150, epoch: 2, batch: 1150, loss: 0.31112, acc: 0.83543
    global step 5160, epoch: 2, batch: 1160, loss: 0.89743, acc: 0.83306
    global step 5170, epoch: 2, batch: 1170, loss: 0.66561, acc: 0.83243
    global step 5180, epoch: 2, batch: 1180, loss: 0.17630, acc: 0.83368
    global step 5190, epoch: 2, batch: 1190, loss: 0.10682, acc: 0.83385
    global step 5200, epoch: 2, batch: 1200, loss: 0.55633, acc: 0.83375
    eval loss: 0.75651, accu: 0.78090
    global step 5210, epoch: 2, batch: 1210, loss: 0.44363, acc: 0.83000
    global step 5220, epoch: 2, batch: 1220, loss: 1.03163, acc: 0.83000
    global step 5230, epoch: 2, batch: 1230, loss: 0.96529, acc: 0.82667
    global step 5240, epoch: 2, batch: 1240, loss: 0.45936, acc: 0.84500
    global step 5250, epoch: 2, batch: 1250, loss: 0.52021, acc: 0.84600
    global step 5260, epoch: 2, batch: 1260, loss: 0.35852, acc: 0.85000
    global step 5270, epoch: 2, batch: 1270, loss: 1.08419, acc: 0.84714
    global step 5280, epoch: 2, batch: 1280, loss: 0.24065, acc: 0.84125
    global step 5290, epoch: 2, batch: 1290, loss: 1.38929, acc: 0.82778
    global step 5300, epoch: 2, batch: 1300, loss: 0.44371, acc: 0.82600
    global step 5310, epoch: 2, batch: 1310, loss: 0.77005, acc: 0.82727
    global step 5320, epoch: 2, batch: 1320, loss: 1.51659, acc: 0.82667
    global step 5330, epoch: 2, batch: 1330, loss: 0.75544, acc: 0.83000
    global step 5340, epoch: 2, batch: 1340, loss: 1.84742, acc: 0.82929
    global step 5350, epoch: 2, batch: 1350, loss: 0.73691, acc: 0.83000
    global step 5360, epoch: 2, batch: 1360, loss: 1.64820, acc: 0.82625
    global step 5370, epoch: 2, batch: 1370, loss: 0.68549, acc: 0.82647
    global step 5380, epoch: 2, batch: 1380, loss: 0.17823, acc: 0.82667
    global step 5390, epoch: 2, batch: 1390, loss: 0.31324, acc: 0.82632
    global step 5400, epoch: 2, batch: 1400, loss: 0.27954, acc: 0.82450
    global step 5410, epoch: 2, batch: 1410, loss: 1.13794, acc: 0.82095
    global step 5420, epoch: 2, batch: 1420, loss: 0.68098, acc: 0.82136
    global step 5430, epoch: 2, batch: 1430, loss: 0.76110, acc: 0.81826
    global step 5440, epoch: 2, batch: 1440, loss: 0.80570, acc: 0.81917
    global step 5450, epoch: 2, batch: 1450, loss: 0.71341, acc: 0.81880
    global step 5460, epoch: 2, batch: 1460, loss: 0.24832, acc: 0.82000
    global step 5470, epoch: 2, batch: 1470, loss: 0.41555, acc: 0.82000
    global step 5480, epoch: 2, batch: 1480, loss: 0.05647, acc: 0.82143
    global step 5490, epoch: 2, batch: 1490, loss: 0.47115, acc: 0.82207
    global step 5500, epoch: 2, batch: 1500, loss: 0.49772, acc: 0.82367
    global step 5510, epoch: 2, batch: 1510, loss: 0.50017, acc: 0.82161
    global step 5520, epoch: 2, batch: 1520, loss: 0.61112, acc: 0.82188
    global step 5530, epoch: 2, batch: 1530, loss: 0.15945, acc: 0.82273
    global step 5540, epoch: 2, batch: 1540, loss: 0.59180, acc: 0.82176
    global step 5550, epoch: 2, batch: 1550, loss: 0.67793, acc: 0.82429
    global step 5560, epoch: 2, batch: 1560, loss: 0.51794, acc: 0.82417
    global step 5570, epoch: 2, batch: 1570, loss: 1.69199, acc: 0.82135
    global step 5580, epoch: 2, batch: 1580, loss: 0.68098, acc: 0.82132
    global step 5590, epoch: 2, batch: 1590, loss: 0.87821, acc: 0.82051
    global step 5600, epoch: 2, batch: 1600, loss: 0.52868, acc: 0.82050
    eval loss: 0.76201, accu: 0.78220
    global step 5610, epoch: 2, batch: 1610, loss: 0.52000, acc: 0.85000
    global step 5620, epoch: 2, batch: 1620, loss: 0.56391, acc: 0.83500
    global step 5630, epoch: 2, batch: 1630, loss: 0.82265, acc: 0.83000
    global step 5640, epoch: 2, batch: 1640, loss: 0.66400, acc: 0.84250
    global step 5650, epoch: 2, batch: 1650, loss: 0.62139, acc: 0.83800
    global step 5660, epoch: 2, batch: 1660, loss: 0.56339, acc: 0.83500
    global step 5670, epoch: 2, batch: 1670, loss: 0.55239, acc: 0.83571
    global step 5680, epoch: 2, batch: 1680, loss: 0.39884, acc: 0.84000
    global step 5690, epoch: 2, batch: 1690, loss: 0.67032, acc: 0.83556
    global step 5700, epoch: 2, batch: 1700, loss: 0.38294, acc: 0.83400
    global step 5710, epoch: 2, batch: 1710, loss: 0.97052, acc: 0.83182
    global step 5720, epoch: 2, batch: 1720, loss: 0.49821, acc: 0.83167
    global step 5730, epoch: 2, batch: 1730, loss: 0.36122, acc: 0.83692
    global step 5740, epoch: 2, batch: 1740, loss: 0.51186, acc: 0.83071
    global step 5750, epoch: 2, batch: 1750, loss: 0.73529, acc: 0.83000
    global step 5760, epoch: 2, batch: 1760, loss: 0.43825, acc: 0.82688
    global step 5770, epoch: 2, batch: 1770, loss: 0.59186, acc: 0.82882
    global step 5780, epoch: 2, batch: 1780, loss: 0.92491, acc: 0.82944
    global step 5790, epoch: 2, batch: 1790, loss: 0.95238, acc: 0.82737
    global step 5800, epoch: 2, batch: 1800, loss: 0.89862, acc: 0.82600
    global step 5810, epoch: 2, batch: 1810, loss: 0.14473, acc: 0.82476
    global step 5820, epoch: 2, batch: 1820, loss: 0.12216, acc: 0.82455
    global step 5830, epoch: 2, batch: 1830, loss: 0.43940, acc: 0.82478
    global step 5840, epoch: 2, batch: 1840, loss: 0.49325, acc: 0.82458
    global step 5850, epoch: 2, batch: 1850, loss: 0.23962, acc: 0.82680
    global step 5860, epoch: 2, batch: 1860, loss: 0.87867, acc: 0.82808
    global step 5870, epoch: 2, batch: 1870, loss: 0.48089, acc: 0.82815
    global step 5880, epoch: 2, batch: 1880, loss: 0.23508, acc: 0.82964
    global step 5890, epoch: 2, batch: 1890, loss: 1.67811, acc: 0.82655
    global step 5900, epoch: 2, batch: 1900, loss: 0.21461, acc: 0.82567
    global step 5910, epoch: 2, batch: 1910, loss: 1.57195, acc: 0.82419
    global step 5920, epoch: 2, batch: 1920, loss: 0.60658, acc: 0.82406
    global step 5930, epoch: 2, batch: 1930, loss: 1.03500, acc: 0.82333
    global step 5940, epoch: 2, batch: 1940, loss: 0.24280, acc: 0.82441
    global step 5950, epoch: 2, batch: 1950, loss: 0.70115, acc: 0.82486
    global step 5960, epoch: 2, batch: 1960, loss: 0.82099, acc: 0.82306
    global step 5970, epoch: 2, batch: 1970, loss: 0.45625, acc: 0.82243
    global step 5980, epoch: 2, batch: 1980, loss: 0.40437, acc: 0.82342
    global step 5990, epoch: 2, batch: 1990, loss: 0.79250, acc: 0.82282
    global step 6000, epoch: 2, batch: 2000, loss: 0.47273, acc: 0.82300
    eval loss: 0.81271, accu: 0.76960
    global step 6010, epoch: 2, batch: 2010, loss: 0.41518, acc: 0.79000
    global step 6020, epoch: 2, batch: 2020, loss: 0.97882, acc: 0.79000
    global step 6030, epoch: 2, batch: 2030, loss: 0.43279, acc: 0.78667
    global step 6040, epoch: 2, batch: 2040, loss: 0.53852, acc: 0.78500
    global step 6050, epoch: 2, batch: 2050, loss: 0.72811, acc: 0.79200
    global step 6060, epoch: 2, batch: 2060, loss: 0.11131, acc: 0.79667
    global step 6070, epoch: 2, batch: 2070, loss: 0.14525, acc: 0.80286
    global step 6080, epoch: 2, batch: 2080, loss: 0.21187, acc: 0.80125
    global step 6090, epoch: 2, batch: 2090, loss: 0.82994, acc: 0.80556
    global step 6100, epoch: 2, batch: 2100, loss: 0.15451, acc: 0.80800
    global step 6110, epoch: 2, batch: 2110, loss: 1.04543, acc: 0.81364
    global step 6120, epoch: 2, batch: 2120, loss: 0.28362, acc: 0.81417
    global step 6130, epoch: 2, batch: 2130, loss: 0.43697, acc: 0.81000
    global step 6140, epoch: 2, batch: 2140, loss: 0.67458, acc: 0.81214
    global step 6150, epoch: 2, batch: 2150, loss: 0.83663, acc: 0.80867
    global step 6160, epoch: 2, batch: 2160, loss: 1.08376, acc: 0.80563
    global step 6170, epoch: 2, batch: 2170, loss: 1.27169, acc: 0.80941
    global step 6180, epoch: 2, batch: 2180, loss: 0.43977, acc: 0.80611
    global step 6190, epoch: 2, batch: 2190, loss: 0.52765, acc: 0.80316
    global step 6200, epoch: 2, batch: 2200, loss: 0.91976, acc: 0.80400
    global step 6210, epoch: 2, batch: 2210, loss: 0.85271, acc: 0.80571
    global step 6220, epoch: 2, batch: 2220, loss: 1.16101, acc: 0.80545
    global step 6230, epoch: 2, batch: 2230, loss: 0.05351, acc: 0.80348
    global step 6240, epoch: 2, batch: 2240, loss: 0.96241, acc: 0.80500
    global step 6250, epoch: 2, batch: 2250, loss: 0.72340, acc: 0.80680
    global step 6260, epoch: 2, batch: 2260, loss: 0.23115, acc: 0.80692
    global step 6270, epoch: 2, batch: 2270, loss: 0.10803, acc: 0.80593
    global step 6280, epoch: 2, batch: 2280, loss: 0.60556, acc: 0.80821
    global step 6290, epoch: 2, batch: 2290, loss: 0.13401, acc: 0.80793
    global step 6300, epoch: 2, batch: 2300, loss: 0.78663, acc: 0.80733
    global step 6310, epoch: 2, batch: 2310, loss: 0.40138, acc: 0.80677
    global step 6320, epoch: 2, batch: 2320, loss: 0.84347, acc: 0.80594
    global step 6330, epoch: 2, batch: 2330, loss: 0.32016, acc: 0.80576
    global step 6340, epoch: 2, batch: 2340, loss: 0.77912, acc: 0.80559
    global step 6350, epoch: 2, batch: 2350, loss: 0.65549, acc: 0.80714
    global step 6360, epoch: 2, batch: 2360, loss: 0.43104, acc: 0.80750
    global step 6370, epoch: 2, batch: 2370, loss: 0.46983, acc: 0.80622
    global step 6380, epoch: 2, batch: 2380, loss: 2.24663, acc: 0.80526
    global step 6390, epoch: 2, batch: 2390, loss: 0.38449, acc: 0.80410
    global step 6400, epoch: 2, batch: 2400, loss: 1.32033, acc: 0.80200
    eval loss: 0.74671, accu: 0.78120
    global step 6410, epoch: 2, batch: 2410, loss: 1.56318, acc: 0.78000
    global step 6420, epoch: 2, batch: 2420, loss: 1.27341, acc: 0.74000
    global step 6430, epoch: 2, batch: 2430, loss: 0.37353, acc: 0.78333
    global step 6440, epoch: 2, batch: 2440, loss: 1.02573, acc: 0.79500
    global step 6450, epoch: 2, batch: 2450, loss: 0.31506, acc: 0.78400
    global step 6460, epoch: 2, batch: 2460, loss: 1.19571, acc: 0.77833
    global step 6470, epoch: 2, batch: 2470, loss: 1.78793, acc: 0.77571
    global step 6480, epoch: 2, batch: 2480, loss: 1.22453, acc: 0.78000
    global step 6490, epoch: 2, batch: 2490, loss: 0.09109, acc: 0.79000
    global step 6500, epoch: 2, batch: 2500, loss: 0.24010, acc: 0.79400
    global step 6510, epoch: 2, batch: 2510, loss: 0.80421, acc: 0.79545
    global step 6520, epoch: 2, batch: 2520, loss: 0.61478, acc: 0.79583
    global step 6530, epoch: 2, batch: 2530, loss: 0.60652, acc: 0.79538
    global step 6540, epoch: 2, batch: 2540, loss: 0.45427, acc: 0.79500
    global step 6550, epoch: 2, batch: 2550, loss: 0.85115, acc: 0.79733
    global step 6560, epoch: 2, batch: 2560, loss: 0.73340, acc: 0.80063
    global step 6570, epoch: 2, batch: 2570, loss: 0.17089, acc: 0.80529
    global step 6580, epoch: 2, batch: 2580, loss: 0.48796, acc: 0.80556
    global step 6590, epoch: 2, batch: 2590, loss: 1.28656, acc: 0.80737
    global step 6600, epoch: 2, batch: 2600, loss: 0.45253, acc: 0.81100
    global step 6610, epoch: 2, batch: 2610, loss: 0.46935, acc: 0.81333
    global step 6620, epoch: 2, batch: 2620, loss: 0.79421, acc: 0.81091
    global step 6630, epoch: 2, batch: 2630, loss: 0.54622, acc: 0.81043
    global step 6640, epoch: 2, batch: 2640, loss: 0.15792, acc: 0.80958
    global step 6650, epoch: 2, batch: 2650, loss: 0.68197, acc: 0.81160
    global step 6660, epoch: 2, batch: 2660, loss: 0.64474, acc: 0.81192
    global step 6670, epoch: 2, batch: 2670, loss: 1.42799, acc: 0.81259
    global step 6680, epoch: 2, batch: 2680, loss: 0.42746, acc: 0.81464
    global step 6690, epoch: 2, batch: 2690, loss: 0.12908, acc: 0.81690
    global step 6700, epoch: 2, batch: 2700, loss: 0.40657, acc: 0.81667
    global step 6710, epoch: 2, batch: 2710, loss: 1.29741, acc: 0.81710
    global step 6720, epoch: 2, batch: 2720, loss: 0.67140, acc: 0.81719
    global step 6730, epoch: 2, batch: 2730, loss: 0.32235, acc: 0.81939
    global step 6740, epoch: 2, batch: 2740, loss: 0.66337, acc: 0.81912
    global step 6750, epoch: 2, batch: 2750, loss: 0.27968, acc: 0.81886
    global step 6760, epoch: 2, batch: 2760, loss: 0.31811, acc: 0.81889
    global step 6770, epoch: 2, batch: 2770, loss: 0.27568, acc: 0.81811
    global step 6780, epoch: 2, batch: 2780, loss: 1.02823, acc: 0.81605
    global step 6790, epoch: 2, batch: 2790, loss: 1.19115, acc: 0.81667
    global step 6800, epoch: 2, batch: 2800, loss: 0.51833, acc: 0.81650
    eval loss: 0.75875, accu: 0.78300
    global step 6810, epoch: 2, batch: 2810, loss: 0.12134, acc: 0.86000
    global step 6820, epoch: 2, batch: 2820, loss: 0.73785, acc: 0.82000
    global step 6830, epoch: 2, batch: 2830, loss: 1.17020, acc: 0.84333
    global step 6840, epoch: 2, batch: 2840, loss: 0.17256, acc: 0.84500
    global step 6850, epoch: 2, batch: 2850, loss: 0.70845, acc: 0.84000
    global step 6860, epoch: 2, batch: 2860, loss: 0.67514, acc: 0.83167
    global step 6870, epoch: 2, batch: 2870, loss: 0.20759, acc: 0.82143
    global step 6880, epoch: 2, batch: 2880, loss: 0.62639, acc: 0.82250
    global step 6890, epoch: 2, batch: 2890, loss: 0.88903, acc: 0.82333
    global step 6900, epoch: 2, batch: 2900, loss: 0.41623, acc: 0.82500
    global step 6910, epoch: 2, batch: 2910, loss: 0.76679, acc: 0.82818
    global step 6920, epoch: 2, batch: 2920, loss: 0.74311, acc: 0.82583
    global step 6930, epoch: 2, batch: 2930, loss: 0.10581, acc: 0.82615
    global step 6940, epoch: 2, batch: 2940, loss: 0.14386, acc: 0.82786
    global step 6950, epoch: 2, batch: 2950, loss: 1.69935, acc: 0.82800
    global step 6960, epoch: 2, batch: 2960, loss: 0.15626, acc: 0.82563
    global step 6970, epoch: 2, batch: 2970, loss: 1.19000, acc: 0.82353
    global step 6980, epoch: 2, batch: 2980, loss: 1.29636, acc: 0.82056
    global step 6990, epoch: 2, batch: 2990, loss: 0.34807, acc: 0.81789
    global step 7000, epoch: 2, batch: 3000, loss: 1.00769, acc: 0.81700
    global step 7010, epoch: 2, batch: 3010, loss: 0.16379, acc: 0.81571
    global step 7020, epoch: 2, batch: 3020, loss: 0.76439, acc: 0.81682
    global step 7030, epoch: 2, batch: 3030, loss: 0.41939, acc: 0.81565
    global step 7040, epoch: 2, batch: 3040, loss: 0.87326, acc: 0.81625
    global step 7050, epoch: 2, batch: 3050, loss: 0.80157, acc: 0.81760
    global step 7060, epoch: 2, batch: 3060, loss: 0.77614, acc: 0.81808
    global step 7070, epoch: 2, batch: 3070, loss: 0.87577, acc: 0.81778
    global step 7080, epoch: 2, batch: 3080, loss: 0.71324, acc: 0.81714
    global step 7090, epoch: 2, batch: 3090, loss: 0.31629, acc: 0.81862
    global step 7100, epoch: 2, batch: 3100, loss: 0.46057, acc: 0.82033
    global step 7110, epoch: 2, batch: 3110, loss: 0.40962, acc: 0.82032
    global step 7120, epoch: 2, batch: 3120, loss: 0.78300, acc: 0.82031
    global step 7130, epoch: 2, batch: 3130, loss: 1.74201, acc: 0.82000
    global step 7140, epoch: 2, batch: 3140, loss: 0.65623, acc: 0.82000
    global step 7150, epoch: 2, batch: 3150, loss: 0.38355, acc: 0.82229
    global step 7160, epoch: 2, batch: 3160, loss: 0.50866, acc: 0.82361
    global step 7170, epoch: 2, batch: 3170, loss: 0.46543, acc: 0.82459
    global step 7180, epoch: 2, batch: 3180, loss: 0.49079, acc: 0.82395
    global step 7190, epoch: 2, batch: 3190, loss: 0.17288, acc: 0.82487
    global step 7200, epoch: 2, batch: 3200, loss: 0.42407, acc: 0.82625
    eval loss: 0.78179, accu: 0.77830
    global step 7210, epoch: 2, batch: 3210, loss: 0.47989, acc: 0.77000
    global step 7220, epoch: 2, batch: 3220, loss: 0.35924, acc: 0.80500
    global step 7230, epoch: 2, batch: 3230, loss: 0.61990, acc: 0.81667
    global step 7240, epoch: 2, batch: 3240, loss: 0.41747, acc: 0.82250
    global step 7250, epoch: 2, batch: 3250, loss: 0.64285, acc: 0.82800
    global step 7260, epoch: 2, batch: 3260, loss: 1.30249, acc: 0.83500
    global step 7270, epoch: 2, batch: 3270, loss: 1.76368, acc: 0.83286
    global step 7280, epoch: 2, batch: 3280, loss: 1.34486, acc: 0.82875
    global step 7290, epoch: 2, batch: 3290, loss: 1.04235, acc: 0.82111
    global step 7300, epoch: 2, batch: 3300, loss: 0.88595, acc: 0.81400
    global step 7310, epoch: 2, batch: 3310, loss: 1.11462, acc: 0.81000
    global step 7320, epoch: 2, batch: 3320, loss: 0.56159, acc: 0.80167
    global step 7330, epoch: 2, batch: 3330, loss: 0.26690, acc: 0.80385
    global step 7340, epoch: 2, batch: 3340, loss: 0.62308, acc: 0.80071
    global step 7350, epoch: 2, batch: 3350, loss: 0.40655, acc: 0.79800
    global step 7360, epoch: 2, batch: 3360, loss: 0.70135, acc: 0.80188
    global step 7370, epoch: 2, batch: 3370, loss: 1.53233, acc: 0.80588
    global step 7380, epoch: 2, batch: 3380, loss: 0.50310, acc: 0.80667
    global step 7390, epoch: 2, batch: 3390, loss: 0.41279, acc: 0.80632
    global step 7400, epoch: 2, batch: 3400, loss: 0.54257, acc: 0.80550
    global step 7410, epoch: 2, batch: 3410, loss: 0.25066, acc: 0.80714
    global step 7420, epoch: 2, batch: 3420, loss: 0.93410, acc: 0.80591
    global step 7430, epoch: 2, batch: 3430, loss: 1.49853, acc: 0.80870
    global step 7440, epoch: 2, batch: 3440, loss: 0.95646, acc: 0.80792
    global step 7450, epoch: 2, batch: 3450, loss: 1.02705, acc: 0.80720
    global step 7460, epoch: 2, batch: 3460, loss: 1.23192, acc: 0.80615
    global step 7470, epoch: 2, batch: 3470, loss: 1.26025, acc: 0.79889
    global step 7480, epoch: 2, batch: 3480, loss: 1.84904, acc: 0.78000
    global step 7490, epoch: 2, batch: 3490, loss: 1.45704, acc: 0.76655
    global step 7500, epoch: 2, batch: 3500, loss: 2.69512, acc: 0.75333
    global step 7510, epoch: 2, batch: 3510, loss: 3.60298, acc: 0.73710
    global step 7520, epoch: 2, batch: 3520, loss: 3.00903, acc: 0.71875
    global step 7530, epoch: 2, batch: 3530, loss: 2.34122, acc: 0.70424
    global step 7540, epoch: 2, batch: 3540, loss: 3.09917, acc: 0.68765
    global step 7550, epoch: 2, batch: 3550, loss: 3.13079, acc: 0.67200
    global step 7560, epoch: 2, batch: 3560, loss: 3.26554, acc: 0.65861
    global step 7570, epoch: 2, batch: 3570, loss: 3.10390, acc: 0.64622
    global step 7580, epoch: 2, batch: 3580, loss: 3.14596, acc: 0.63342
    global step 7590, epoch: 2, batch: 3590, loss: 2.75602, acc: 0.62103
    global step 7600, epoch: 2, batch: 3600, loss: 2.79565, acc: 0.61200
    eval loss: 3.06398, accu: 0.22050
    global step 7610, epoch: 2, batch: 3610, loss: 3.06742, acc: 0.23000
    global step 7620, epoch: 2, batch: 3620, loss: 3.08773, acc: 0.23500
    global step 7630, epoch: 2, batch: 3630, loss: 3.40460, acc: 0.21333
    global step 7640, epoch: 2, batch: 3640, loss: 2.75078, acc: 0.22000
    global step 7650, epoch: 2, batch: 3650, loss: 2.71861, acc: 0.22200
    global step 7660, epoch: 2, batch: 3660, loss: 3.10492, acc: 0.22000
    global step 7670, epoch: 2, batch: 3670, loss: 3.05878, acc: 0.21429
    global step 7680, epoch: 2, batch: 3680, loss: 3.20089, acc: 0.21625
    global step 7690, epoch: 2, batch: 3690, loss: 3.53724, acc: 0.21444
    global step 7700, epoch: 2, batch: 3700, loss: 3.33438, acc: 0.20700
    global step 7710, epoch: 2, batch: 3710, loss: 2.60782, acc: 0.20182
    global step 7720, epoch: 2, batch: 3720, loss: 3.02315, acc: 0.20333
    global step 7730, epoch: 2, batch: 3730, loss: 2.83833, acc: 0.20462
    global step 7740, epoch: 2, batch: 3740, loss: 3.30064, acc: 0.20143
    global step 7750, epoch: 2, batch: 3750, loss: 3.25825, acc: 0.20667
    global step 7760, epoch: 2, batch: 3760, loss: 3.15886, acc: 0.20688
    global step 7770, epoch: 2, batch: 3770, loss: 2.67722, acc: 0.20941
    global step 7780, epoch: 2, batch: 3780, loss: 3.76232, acc: 0.20444
    global step 7790, epoch: 2, batch: 3790, loss: 2.75780, acc: 0.20368
    global step 7800, epoch: 2, batch: 3800, loss: 2.75351, acc: 0.20550
    global step 7810, epoch: 2, batch: 3810, loss: 3.47542, acc: 0.20524
    global step 7820, epoch: 2, batch: 3820, loss: 3.07591, acc: 0.20500
    global step 7830, epoch: 2, batch: 3830, loss: 2.46672, acc: 0.20435
    global step 7840, epoch: 2, batch: 3840, loss: 3.16461, acc: 0.20542
    global step 7850, epoch: 2, batch: 3850, loss: 3.15110, acc: 0.20600
    global step 7860, epoch: 2, batch: 3860, loss: 1.89892, acc: 0.21000
    global step 7870, epoch: 2, batch: 3870, loss: 2.86100, acc: 0.20926
    global step 7880, epoch: 2, batch: 3880, loss: 3.05075, acc: 0.21000
    global step 7890, epoch: 2, batch: 3890, loss: 3.33859, acc: 0.21069
    global step 7900, epoch: 2, batch: 3900, loss: 2.74557, acc: 0.21067
    global step 7910, epoch: 2, batch: 3910, loss: 3.26475, acc: 0.21065
    global step 7920, epoch: 2, batch: 3920, loss: 2.97915, acc: 0.21188
    global step 7930, epoch: 2, batch: 3930, loss: 3.29079, acc: 0.21242
    global step 7940, epoch: 2, batch: 3940, loss: 3.03578, acc: 0.21059
    global step 7950, epoch: 2, batch: 3950, loss: 3.22661, acc: 0.21029
    global step 7960, epoch: 2, batch: 3960, loss: 3.15023, acc: 0.20917
    global step 7970, epoch: 2, batch: 3970, loss: 3.08591, acc: 0.20811
    global step 7980, epoch: 2, batch: 3980, loss: 2.62390, acc: 0.21000
    global step 7990, epoch: 2, batch: 3990, loss: 3.15164, acc: 0.20897
    global step 8000, epoch: 2, batch: 4000, loss: 3.19271, acc: 0.21050
    eval loss: 3.05533, accu: 0.22050
    global step 8010, epoch: 3, batch: 10, loss: 2.93468, acc: 0.25000
    global step 8020, epoch: 3, batch: 20, loss: 3.17175, acc: 0.24500
    global step 8030, epoch: 3, batch: 30, loss: 3.39411, acc: 0.23333
    global step 8040, epoch: 3, batch: 40, loss: 2.61441, acc: 0.24000
    global step 8050, epoch: 3, batch: 50, loss: 3.42808, acc: 0.22000
    global step 8060, epoch: 3, batch: 60, loss: 3.07453, acc: 0.21833
    global step 8070, epoch: 3, batch: 70, loss: 3.75857, acc: 0.21571
    global step 8080, epoch: 3, batch: 80, loss: 2.97565, acc: 0.21750
    global step 8090, epoch: 3, batch: 90, loss: 3.41039, acc: 0.21778
    global step 8100, epoch: 3, batch: 100, loss: 2.92406, acc: 0.22200
    global step 8110, epoch: 3, batch: 110, loss: 2.72436, acc: 0.22182
    global step 8120, epoch: 3, batch: 120, loss: 3.07305, acc: 0.21750
    global step 8130, epoch: 3, batch: 130, loss: 3.64916, acc: 0.21615
    global step 8140, epoch: 3, batch: 140, loss: 2.45027, acc: 0.22071
    global step 8150, epoch: 3, batch: 150, loss: 2.32821, acc: 0.22200
    global step 8160, epoch: 3, batch: 160, loss: 2.50674, acc: 0.22625
    global step 8170, epoch: 3, batch: 170, loss: 2.98994, acc: 0.22353
    global step 8180, epoch: 3, batch: 180, loss: 2.78187, acc: 0.22889
    global step 8190, epoch: 3, batch: 190, loss: 3.34695, acc: 0.22579
    global step 8200, epoch: 3, batch: 200, loss: 3.60488, acc: 0.22300
    global step 8210, epoch: 3, batch: 210, loss: 3.10538, acc: 0.22143
    global step 8220, epoch: 3, batch: 220, loss: 3.47500, acc: 0.22364
    global step 8230, epoch: 3, batch: 230, loss: 3.21620, acc: 0.22435
    global step 8240, epoch: 3, batch: 240, loss: 2.99791, acc: 0.22542
    global step 8250, epoch: 3, batch: 250, loss: 3.29019, acc: 0.22680
    global step 8260, epoch: 3, batch: 260, loss: 3.08440, acc: 0.22731
    global step 8270, epoch: 3, batch: 270, loss: 2.70554, acc: 0.22963
    global step 8280, epoch: 3, batch: 280, loss: 3.18647, acc: 0.23000
    global step 8290, epoch: 3, batch: 290, loss: 2.85845, acc: 0.22897
    global step 8300, epoch: 3, batch: 300, loss: 3.12906, acc: 0.22833
    global step 8310, epoch: 3, batch: 310, loss: 2.83453, acc: 0.22903
    global step 8320, epoch: 3, batch: 320, loss: 3.02896, acc: 0.22719
    global step 8330, epoch: 3, batch: 330, loss: 3.29427, acc: 0.22667
    global step 8340, epoch: 3, batch: 340, loss: 3.10322, acc: 0.22500
    global step 8350, epoch: 3, batch: 350, loss: 3.25135, acc: 0.22286
    global step 8360, epoch: 3, batch: 360, loss: 2.95804, acc: 0.22139
    global step 8370, epoch: 3, batch: 370, loss: 3.00305, acc: 0.22054
    global step 8380, epoch: 3, batch: 380, loss: 3.00939, acc: 0.22079
    global step 8390, epoch: 3, batch: 390, loss: 3.25538, acc: 0.22308
    global step 8400, epoch: 3, batch: 400, loss: 2.82099, acc: 0.22500
    eval loss: 3.09904, accu: 0.22050
    global step 8410, epoch: 3, batch: 410, loss: 3.03722, acc: 0.30000
    global step 8420, epoch: 3, batch: 420, loss: 2.99034, acc: 0.27000
    global step 8430, epoch: 3, batch: 430, loss: 3.53011, acc: 0.24667
    global step 8440, epoch: 3, batch: 440, loss: 3.23101, acc: 0.22500
    global step 8450, epoch: 3, batch: 450, loss: 2.61292, acc: 0.22800
    global step 8460, epoch: 3, batch: 460, loss: 2.87655, acc: 0.23000
    global step 8470, epoch: 3, batch: 470, loss: 2.77331, acc: 0.22429
    global step 8480, epoch: 3, batch: 480, loss: 2.96680, acc: 0.21500
    global step 8490, epoch: 3, batch: 490, loss: 3.37138, acc: 0.21333
    global step 8500, epoch: 3, batch: 500, loss: 4.19576, acc: 0.21400
    global step 8510, epoch: 3, batch: 510, loss: 3.23210, acc: 0.21000
    global step 8520, epoch: 3, batch: 520, loss: 2.97283, acc: 0.20917
    global step 8530, epoch: 3, batch: 530, loss: 3.08502, acc: 0.20769
    global step 8540, epoch: 3, batch: 540, loss: 2.77662, acc: 0.21214
    global step 8550, epoch: 3, batch: 550, loss: 3.18841, acc: 0.21267
    global step 8560, epoch: 3, batch: 560, loss: 2.64816, acc: 0.21063
    global step 8570, epoch: 3, batch: 570, loss: 2.94915, acc: 0.21471
    global step 8580, epoch: 3, batch: 580, loss: 2.71477, acc: 0.21667
    global step 8590, epoch: 3, batch: 590, loss: 3.33645, acc: 0.21474
    global step 8600, epoch: 3, batch: 600, loss: 3.10244, acc: 0.21750
    global step 8610, epoch: 3, batch: 610, loss: 2.93462, acc: 0.21905
    global step 8620, epoch: 3, batch: 620, loss: 3.15820, acc: 0.21636
    global step 8630, epoch: 3, batch: 630, loss: 3.23823, acc: 0.21478
    global step 8640, epoch: 3, batch: 640, loss: 3.05600, acc: 0.21542
    global step 8650, epoch: 3, batch: 650, loss: 2.97864, acc: 0.21400
    global step 8660, epoch: 3, batch: 660, loss: 2.96220, acc: 0.21808
    global step 8670, epoch: 3, batch: 670, loss: 2.54549, acc: 0.21667
    global step 8680, epoch: 3, batch: 680, loss: 3.63373, acc: 0.21750
    global step 8690, epoch: 3, batch: 690, loss: 3.15993, acc: 0.21793
    global step 8700, epoch: 3, batch: 700, loss: 3.10693, acc: 0.21833
    global step 8710, epoch: 3, batch: 710, loss: 2.75067, acc: 0.21806
    global step 8720, epoch: 3, batch: 720, loss: 2.85182, acc: 0.21625
    global step 8730, epoch: 3, batch: 730, loss: 3.69052, acc: 0.21697
    global step 8740, epoch: 3, batch: 740, loss: 2.73857, acc: 0.21794
    global step 8750, epoch: 3, batch: 750, loss: 3.35997, acc: 0.22029
    global step 8760, epoch: 3, batch: 760, loss: 3.40170, acc: 0.21972
    global step 8770, epoch: 3, batch: 770, loss: 3.00354, acc: 0.22027
    global step 8780, epoch: 3, batch: 780, loss: 3.47179, acc: 0.22105
    global step 8790, epoch: 3, batch: 790, loss: 3.36912, acc: 0.22179
    global step 8800, epoch: 3, batch: 800, loss: 3.50122, acc: 0.22250
    eval loss: 3.06095, accu: 0.22050
    global step 8810, epoch: 3, batch: 810, loss: 3.18056, acc: 0.21000
    global step 8820, epoch: 3, batch: 820, loss: 3.22122, acc: 0.20000
    global step 8830, epoch: 3, batch: 830, loss: 3.20884, acc: 0.21667
    global step 8840, epoch: 3, batch: 840, loss: 2.97862, acc: 0.22500
    global step 8850, epoch: 3, batch: 850, loss: 3.22756, acc: 0.23000
    global step 8860, epoch: 3, batch: 860, loss: 3.10388, acc: 0.23000
    global step 8870, epoch: 3, batch: 870, loss: 2.65904, acc: 0.22571
    global step 8880, epoch: 3, batch: 880, loss: 2.35919, acc: 0.22875
    global step 8890, epoch: 3, batch: 890, loss: 2.68343, acc: 0.22889
    global step 8900, epoch: 3, batch: 900, loss: 3.76593, acc: 0.23800
    global step 8910, epoch: 3, batch: 910, loss: 3.65261, acc: 0.23636
    global step 8920, epoch: 3, batch: 920, loss: 2.59790, acc: 0.23417
    global step 8930, epoch: 3, batch: 930, loss: 2.28856, acc: 0.23615
    global step 8940, epoch: 3, batch: 940, loss: 3.33877, acc: 0.23286
    global step 8950, epoch: 3, batch: 950, loss: 3.15151, acc: 0.23200
    global step 8960, epoch: 3, batch: 960, loss: 2.68373, acc: 0.23000
    global step 8970, epoch: 3, batch: 970, loss: 3.14382, acc: 0.23176
    global step 8980, epoch: 3, batch: 980, loss: 3.90098, acc: 0.23111
    global step 8990, epoch: 3, batch: 990, loss: 2.79722, acc: 0.23105
    global step 9000, epoch: 3, batch: 1000, loss: 2.81255, acc: 0.23250
    global step 9010, epoch: 3, batch: 1010, loss: 2.90026, acc: 0.23048
    global step 9020, epoch: 3, batch: 1020, loss: 2.99405, acc: 0.23318
    global step 9030, epoch: 3, batch: 1030, loss: 3.17540, acc: 0.23391
    global step 9040, epoch: 3, batch: 1040, loss: 2.77659, acc: 0.23583
    global step 9050, epoch: 3, batch: 1050, loss: 3.84944, acc: 0.23200
    global step 9060, epoch: 3, batch: 1060, loss: 3.50860, acc: 0.23192
    global step 9070, epoch: 3, batch: 1070, loss: 2.78802, acc: 0.23074
    global step 9080, epoch: 3, batch: 1080, loss: 2.48953, acc: 0.23071
    global step 9090, epoch: 3, batch: 1090, loss: 3.14193, acc: 0.23034
    global step 9100, epoch: 3, batch: 1100, loss: 3.12116, acc: 0.22900
    global step 9110, epoch: 3, batch: 1110, loss: 2.94410, acc: 0.23000
    global step 9120, epoch: 3, batch: 1120, loss: 3.39145, acc: 0.23062
    global step 9130, epoch: 3, batch: 1130, loss: 2.98915, acc: 0.22788
    global step 9140, epoch: 3, batch: 1140, loss: 3.31599, acc: 0.22794
    global step 9150, epoch: 3, batch: 1150, loss: 3.12026, acc: 0.22743
    global step 9160, epoch: 3, batch: 1160, loss: 3.33454, acc: 0.22639
    global step 9170, epoch: 3, batch: 1170, loss: 2.89012, acc: 0.22568
    global step 9180, epoch: 3, batch: 1180, loss: 3.31300, acc: 0.22579
    global step 9190, epoch: 3, batch: 1190, loss: 3.31653, acc: 0.22718
    global step 9200, epoch: 3, batch: 1200, loss: 2.86382, acc: 0.22675
    eval loss: 3.06819, accu: 0.22050
    global step 9210, epoch: 3, batch: 1210, loss: 3.33352, acc: 0.26000
    global step 9220, epoch: 3, batch: 1220, loss: 2.62750, acc: 0.26000
    global step 9230, epoch: 3, batch: 1230, loss: 3.09104, acc: 0.26667
    global step 9240, epoch: 3, batch: 1240, loss: 2.42483, acc: 0.26250
    global step 9250, epoch: 3, batch: 1250, loss: 3.14092, acc: 0.25000
    global step 9260, epoch: 3, batch: 1260, loss: 3.12716, acc: 0.24000
    global step 9270, epoch: 3, batch: 1270, loss: 2.94051, acc: 0.24000
    global step 9280, epoch: 3, batch: 1280, loss: 3.39405, acc: 0.23875
    global step 9290, epoch: 3, batch: 1290, loss: 2.43539, acc: 0.24000
    global step 9300, epoch: 3, batch: 1300, loss: 3.22079, acc: 0.24400
    global step 9310, epoch: 3, batch: 1310, loss: 3.41507, acc: 0.24364
    global step 9320, epoch: 3, batch: 1320, loss: 3.26469, acc: 0.23750
    global step 9330, epoch: 3, batch: 1330, loss: 2.96452, acc: 0.23692
    global step 9340, epoch: 3, batch: 1340, loss: 3.86649, acc: 0.23929
    global step 9350, epoch: 3, batch: 1350, loss: 3.37465, acc: 0.24267
    global step 9360, epoch: 3, batch: 1360, loss: 2.78971, acc: 0.23938
    global step 9370, epoch: 3, batch: 1370, loss: 3.55265, acc: 0.23824
    global step 9380, epoch: 3, batch: 1380, loss: 3.32525, acc: 0.23667
    global step 9390, epoch: 3, batch: 1390, loss: 3.04595, acc: 0.23737
    global step 9400, epoch: 3, batch: 1400, loss: 3.49904, acc: 0.23350
    global step 9410, epoch: 3, batch: 1410, loss: 2.94410, acc: 0.23286
    global step 9420, epoch: 3, batch: 1420, loss: 2.93697, acc: 0.23182
    global step 9430, epoch: 3, batch: 1430, loss: 2.82375, acc: 0.23043
    global step 9440, epoch: 3, batch: 1440, loss: 2.35573, acc: 0.22875
    global step 9450, epoch: 3, batch: 1450, loss: 2.64705, acc: 0.22840
    global step 9460, epoch: 3, batch: 1460, loss: 2.68510, acc: 0.22731
    global step 9470, epoch: 3, batch: 1470, loss: 3.05000, acc: 0.22593
    global step 9480, epoch: 3, batch: 1480, loss: 2.97262, acc: 0.22500
    global step 9490, epoch: 3, batch: 1490, loss: 3.06183, acc: 0.22552
    global step 9500, epoch: 3, batch: 1500, loss: 2.95046, acc: 0.22700
    global step 9510, epoch: 3, batch: 1510, loss: 3.60980, acc: 0.22903
    global step 9520, epoch: 3, batch: 1520, loss: 3.38786, acc: 0.22719
    global step 9530, epoch: 3, batch: 1530, loss: 3.05177, acc: 0.22939
    global step 9540, epoch: 3, batch: 1540, loss: 2.91108, acc: 0.22882
    global step 9550, epoch: 3, batch: 1550, loss: 3.37512, acc: 0.22771
    global step 9560, epoch: 3, batch: 1560, loss: 3.50239, acc: 0.22722
    global step 9570, epoch: 3, batch: 1570, loss: 2.65319, acc: 0.22676
    global step 9580, epoch: 3, batch: 1580, loss: 2.94844, acc: 0.22632
    global step 9590, epoch: 3, batch: 1590, loss: 3.61321, acc: 0.22641
    global step 9600, epoch: 3, batch: 1600, loss: 2.82690, acc: 0.22725
    eval loss: 3.05093, accu: 0.22050



    ---------------------------------------------------------------------------
    
    KeyboardInterrupt                         Traceback (most recent call last)
    
    <ipython-input-21-21d1e9c802a4> in <module>
         10         input_ids, segment_ids, labels = batch
         11         logits = model(input_ids)
    ---> 12         loss = criterion(logits, labels)
         13         probs = F.softmax(logits, axis=1)
         14         correct = metric.compute(probs, labels)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py in __call__(self, *inputs, **kwargs)
        896                 self._built = True
        897 
    --> 898             outputs = self.forward(*inputs, **kwargs)
        899 
        900             for forward_post_hook in self._forward_post_hooks.values():


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/layer/loss.py in forward(self, input, label)
        403             axis=self.axis,
        404             use_softmax=self.use_softmax,
    --> 405             name=self.name)
        406 
        407         return ret


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/nn/functional/loss.py in cross_entropy(input, label, weight, ignore_index, reduction, soft_label, axis, use_softmax, name)
       1390             input, label, 'soft_label', soft_label, 'ignore_index',
       1391             ignore_index, 'numeric_stable_mode', True, 'axis', axis,
    -> 1392             'use_softmax', use_softmax)
       1393 
       1394         if weight is not None:


    KeyboardInterrupt: 


## 5.训练日志
```
eval loss: 1.22675, accu: 0.77660
global step 6810, epoch: 10, batch: 474, loss: 0.01697, acc: 0.99219
global step 6820, epoch: 10, batch: 484, loss: 0.04531, acc: 0.98984
global step 6830, epoch: 10, batch: 494, loss: 0.03325, acc: 0.98854
global step 6840, epoch: 10, batch: 504, loss: 0.04574, acc: 0.98672
global step 6850, epoch: 10, batch: 514, loss: 0.02137, acc: 0.98625
global step 6860, epoch: 10, batch: 524, loss: 0.19356, acc: 0.98516
global step 6870, epoch: 10, batch: 534, loss: 0.03456, acc: 0.98482
global step 6880, epoch: 10, batch: 544, loss: 0.09647, acc: 0.98438
global step 6890, epoch: 10, batch: 554, loss: 0.11611, acc: 0.98351
global step 6900, epoch: 10, batch: 564, loss: 0.05723, acc: 0.98344
global step 6910, epoch: 10, batch: 574, loss: 0.00518, acc: 0.98310
global step 6920, epoch: 10, batch: 584, loss: 0.01201, acc: 0.98281
global step 6930, epoch: 10, batch: 594, loss: 0.07870, acc: 0.98221
global step 6940, epoch: 10, batch: 604, loss: 0.01748, acc: 0.98237
global step 6950, epoch: 10, batch: 614, loss: 0.01542, acc: 0.98208
global step 6960, epoch: 10, batch: 624, loss: 0.01469, acc: 0.98184
global step 6970, epoch: 10, batch: 634, loss: 0.07767, acc: 0.98189
global step 6980, epoch: 10, batch: 644, loss: 0.01516, acc: 0.98186
global step 6990, epoch: 10, batch: 654, loss: 0.02567, acc: 0.98125
global step 7000, epoch: 10, batch: 664, loss: 0.09072, acc: 0.98102
global step 7010, epoch: 10, batch: 674, loss: 0.07557, acc: 0.98080
global step 7020, epoch: 10, batch: 684, loss: 0.13695, acc: 0.98047
global step 7030, epoch: 10, batch: 694, loss: 0.09411, acc: 0.98016
global step 7040, epoch: 10, batch: 704, loss: 0.10656, acc: 0.98007
```

![](https://ai-studio-static-online.cdn.bcebos.com/39efd5a5f54f4ac3ab11eb4ff208eaebd0a3aa0203d64ce783e13b1194e146b7)



```python
tokenizer.save_pretrained(save_dir)
```

# 五、模型预测

## 1.test数据处理


```python
import pandas as pd
from paddlenlp.datasets import load_dataset
import paddlenlp as ppnlp
from functools import partial
from paddlenlp.data import Stack, Tuple, Pad
from utils import  convert_example, create_dataloader
import os
import numpy as np
import paddle
import paddle.nn.functional as F
```


```python
test = pd.read_csv('dataset/test.csv', sep='\t')
sub = pd.read_csv('dataset/Datawhale_学术论文分类_数据集/sample_submit.csv')


train = pd.read_csv('dataset/train.csv', sep='\t')
label_id2cate = dict(enumerate(train.categories.unique()))
label_cate2id = {value: key for key, value in label_id2cate.items()}
```


```python
# 拼接title与abstract
test['text'] = test['title'] + ' ' + test['abstract']
```


```python
print(label_id2cate)
print(label_cate2id)
print(len(label_cate2id))
```

    {0: 'cs.CL', 1: 'cs.NE', 2: 'cs.DL', 3: 'cs.CV', 4: 'cs.LG', 5: 'cs.DS', 6: 'cs.IR', 7: 'cs.RO', 8: 'cs.DM', 9: 'cs.CR', 10: 'cs.AR', 11: 'cs.NI', 12: 'cs.AI', 13: 'cs.SE', 14: 'cs.CG', 15: 'cs.LO', 16: 'cs.SY', 17: 'cs.GR', 18: 'cs.PL', 19: 'cs.SI', 20: 'cs.OH', 21: 'cs.HC', 22: 'cs.MA', 23: 'cs.GT', 24: 'cs.ET', 25: 'cs.FL', 26: 'cs.CC', 27: 'cs.DB', 28: 'cs.DC', 29: 'cs.CY', 30: 'cs.CE', 31: 'cs.MM', 32: 'cs.NA', 33: 'cs.PF', 34: 'cs.OS', 35: 'cs.SD', 36: 'cs.SC', 37: 'cs.MS', 38: 'cs.GL'}
    {'cs.CL': 0, 'cs.NE': 1, 'cs.DL': 2, 'cs.CV': 3, 'cs.LG': 4, 'cs.DS': 5, 'cs.IR': 6, 'cs.RO': 7, 'cs.DM': 8, 'cs.CR': 9, 'cs.AR': 10, 'cs.NI': 11, 'cs.AI': 12, 'cs.SE': 13, 'cs.CG': 14, 'cs.LO': 15, 'cs.SY': 16, 'cs.GR': 17, 'cs.PL': 18, 'cs.SI': 19, 'cs.OH': 20, 'cs.HC': 21, 'cs.MA': 22, 'cs.GT': 23, 'cs.ET': 24, 'cs.FL': 25, 'cs.CC': 26, 'cs.DB': 27, 'cs.DC': 28, 'cs.CY': 29, 'cs.CE': 30, 'cs.MM': 31, 'cs.NA': 32, 'cs.PF': 33, 'cs.OS': 34, 'cs.SD': 35, 'cs.SC': 36, 'cs.MS': 37, 'cs.GL': 38}
    39



```python
# read test data
def read_test(pd_data):
    for index, item in pd_data.iterrows():       
        yield {'text': item['text'], 'label': 0, 'qid': item['paperid'].strip('test_')}
```


```python
test_ds =  load_dataset(read_test, pd_data=test,lazy=False)
```


```python
for i in range(5):
    print(test_ds[i])
```


```python
print(len(test_ds))
```


```python
import paddlenlp as ppnlp

from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
# 指定模型名称，一键加载模型
model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en", num_classes=39)
# 同样地，通过指定模型名称一键加载对应的Tokenizer，用于处理文本数据，如切分token，转token_id等。
tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en")
```

    [2021-07-25 23:03:58,725] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_2.0_large_en/skep_ernie_2.0_large_en.pdparams
    [2021-07-25 23:04:04,013] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_2.0_large_en/skep_ernie_2.0_large_en.vocab.txt



```python
max_seq_length = 300

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    is_test=True)
```


```python
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
```


```python
test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=10,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

## 2.载入模型


```python
# 根据实际运行情况，更换加载的参数路径
import os
import paddle

params_path = 'checkpoint/model_state.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
```

    Loaded parameters from checkpoint/model_state.pdparams


## 3.预测


```python
import os
from functools import partial
import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from utils import create_dataloader

results = []
# 切换model模型为评估模式，关闭dropout等随机因素
model.eval()
for batch in test_data_loader:
    input_ids, token_type_ids, qids = batch
    # 喂数据给模型
    logits = model(input_ids, token_type_ids)
    # 预测分类
    probs = F.softmax(logits, axis=-1)
    idx = paddle.argmax(probs, axis=1).numpy()
    idx = idx.tolist()
    labels = [label_id2cate[i] for i in idx]
    qids = qids.numpy().tolist()
    results.extend( labels)
```


```python
print(results[:5])
print(len(results))
```

## 4.保存提交


```python
sub = pd.read_csv('dataset/Datawhale_学术论文分类_数据集/sample_submit.csv')
sub['categories'] = results
sub.to_csv('submit.csv', index=False)
```


```python
!zip -qr result.zip submit.csv
```

# 六、提交结果

提交结果第14名

![](https://ai-studio-static-online.cdn.bcebos.com/dde0047b3267489fb774b11212b7653ef2e914a8d5cf4690b0a585d94c36bb0a)

