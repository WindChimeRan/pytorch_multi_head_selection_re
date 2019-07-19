# "Joint entity recognition and relation extraction as a multi-head selection problem" (Expert Syst. Appl, 2018)

[paper](https://arxiv.org/abs/1804.07847)

[official tensorflow version](https://github.com/bekou/multihead_joint_entity_relation_extraction)

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
# Result

## Chinese
Training speed: 10min/epoch

|  | precision | recall | f1 |
| ------ | ------ | ------ | ------ |
| Ours (dev) | 0.7443 | 0.6960 | 0.7194 |
| Winner (test) | 0.8975 |0.8886 | 0.893 |

## CoNLL04

|  | precision | recall | f1 |
| ------ | ------ | ------ | ------ |
| Ours (test) | 0.6531 | 0.3153 | 0.4252 |
| Official (test) | 0.6375 |0.6043 | 0.6204 |

The official score is suspicious. They seems use less strict evaluation.

We use the strictest setting: a triplet is correct only if the relation and all the tokens of head and tail are correct. 




# PRs welcome

Current status
* No hyperparameter tuning
* No pretrained embedding
* No bert embedding
* No word-char embedding

Need more datasets and compared models.
