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

On going

# Run
```shell
python main.py --mode preprocessing --exp_name chinese_selection_re
python main.py --mode train --exp_name chinese_selection_re 
python main.py --mode evaluation --exp_name chinese_selection_re

```
# Result

Training speed: 10min/epoch

|  | precision | recall | f1 |
| ------ | ------ | ------ | ------ |
|Ours (dev) | 0.7443 | 0.6960 | 0.7194 |
| Winner (test) | 0.8975 |0.8886 | 0.893 |


# PRs welcome

Current status
* No hyperparameter tuning
* No pretrained embedding
* No bert embedding
* No word-char embedding

Need more datasets and compared models.
