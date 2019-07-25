# "Joint entity recognition and relation extraction as a multi-head selection problem" (Expert Syst. Appl, 2018)

[paper](https://arxiv.org/abs/1804.07847)

[official tensorflow version](https://github.com/bekou/multihead_joint_entity_relation_extraction)

This model is extreamly useful for real-world RE usage. I originally reimplemented for a competition (Chinese IE). I will add CoNLL04 dataset and BERT model.

# Requirement

* python 3.7
* pytorch 1.10

# Dataset

## Chinese IE
Chinese Information Extraction Competition [link](http://lic2019.ccf.org.cn/kg)

**Unzip \*.json into ./raw_data/chinese/**

## CoNLL04

We use the data processed by official version.

**already in ./raw_data/CoNLL04/**


# Run
```shell
python main.py --mode preprocessing --exp_name chinese_selection_re
python main.py --mode train --exp_name chinese_selection_re 
python main.py --mode evaluation --exp_name chinese_selection_re
```

If you want to try other experiments:

set **exp_name** as **conll_selection_re** or **conll_bert_re**



# Result

## Chinese
Training speed: 10min/epoch

|  | precision | recall | f1 |
| ------ | ------ | ------ | ------ |
| Ours (dev) | 0.7443 | 0.6960 | 0.7194 |
| Winner (test) | 0.8975 |0.8886 | 0.893 |

## CoNLL04
Test set:

|  | precision | recall | f1 |
| ------ | ------ | ------ | ------ |
| Ours (LSTM) | 0.6531 | 0.3153 | 0.4252 |
| Ours (BERT-freeze) | 0.5233 | 0.4975 | 0.5101 |
| Official | 0.6375 |0.6043 | 0.6204 |

We use the strictest setting: a triplet is correct only if the relation and all the tokens of head and tail are correct. 


# Details

The model was originally used for Chinese IE, thus, it's a bit different from the official paper:

They use pretrained char-word embedding while we use word embedding initialized randomly; they use 3-layer LSTM while we use 1-layer LSTM.

# TODO

* Tune the hyperparameters for CoNLL04
