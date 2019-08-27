import os
import json

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from pytorch_transformers import *


class Selection_Dataset(Dataset):
    def __init__(self, hyper, dataset, type='train|eval'):
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.type = type

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r'))

        self.selection_list = []
        self.text_list = []
        self.bio_list = []
        self.spo_list = []
        self.jobid_list = []

        # for bert only
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')

        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip("\n")
            instance = json.loads(line)
            self.text_list.append(instance['text'])
            if self.type == 'predict':
                continue

            self.selection_list.append(instance['selection'])
            self.bio_list.append(instance['bio'])
            self.spo_list.append(instance['spo_list'])
            self.jobid_list.append(instance.get('jobid', 0))

    def __getitem__(self, index):
        text = self.text_list[index]
        if self.type == 'predict':
            tokens_id = self.text2tensor(text)
            return tokens_id, len(text), text

        selection = self.selection_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]
        jobid = self.jobid_list[index]
        if self.hyper.cell_name == 'bert':
            text, bio, selection = self.pad_bert(text, bio, selection)
            tokens_id = torch.tensor(
                self.bert_tokenizer.convert_tokens_to_ids(text))
        else:
            tokens_id = self.text2tensor(text)
        bio_id = self.bio2tensor(bio)
        selection_id = self.selection2tensor(text, selection)

        return tokens_id, bio_id, selection_id, len(text), spo, text, bio, jobid

    def __len__(self):
        return len(self.text_list)

    def pad_bert(self, text: List[str], bio: List[str], selection: List[Dict[str, int]]) -> Tuple[List[str], List[str], Dict[str, int]]:
        # for [CLS] and [SEP]
        text = ['[CLS]'] + text + ['[SEP]']
        bio = ['O'] + bio + ['O']
        selection = [{'subject': triplet['subject'] + 1, 'object': triplet['object'] +
                      1, 'predicate': triplet['predicate']} for triplet in selection]
        assert len(text) <= self.hyper.max_text_len
        text = text + ['[PAD]'] * (self.hyper.max_text_len - len(text))
        return text, bio, selection

    def text2tensor(self, text: List[str]) -> torch.tensor:
        # TODO: tokenizer
        oov = self.word_vocab['oov']
        padded_list = list(map(lambda x: self.word_vocab.get(x, oov), text))
        padded_list.extend([self.word_vocab['<pad>']] *
                           (self.hyper.max_text_len - len(text)))
        return torch.tensor(padded_list)

    def bio2tensor(self, bio):
        # here we pad bio with "O". Then, in our model, we will mask this "O" padding.
        # in multi-head selection, we will use "<pad>" token embedding instead.
        padded_list = list(map(lambda x: self.bio_vocab[x], bio))
        padded_list.extend([self.bio_vocab['O']] *
                           (self.hyper.max_text_len - len(bio)))
        return torch.tensor(padded_list)

    def selection2tensor(self, text, selection):
        # s p o
        result = torch.zeros(
            (self.hyper.max_text_len, len(self.relation_vocab),
             self.hyper.max_text_len))
        NA = self.relation_vocab['N']
        result[:, NA, :] = 1
        for triplet in selection:

            object = triplet['object']
            subject = triplet['subject']
            predicate = triplet['predicate']

            result[subject, predicate, object] = 1
            result[subject, NA, object] = 0

        return result


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        # tokens_id, bio_id, selection_id, spo, text, bio

        self.tokens_id = pad_sequence(transposed_data[0], batch_first=True)
        self.bio_id = pad_sequence(transposed_data[1], batch_first=True)
        self.selection_id = torch.stack(transposed_data[2], 0)

        self.length = transposed_data[3]

        self.spo_gold = transposed_data[4]
        self.text = transposed_data[5]
        self.bio = transposed_data[6]
        self.jobid = transposed_data[7]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        self.bio_id = self.bio_id.pin_memory()
        self.selection_id = self.selection_id.pin_memory()
        return self


def collate_fn(batch):
    return Batch_reader(batch)


Selection_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)


# for predict ...
class Batch_reader_predict(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        # tokens_id, len(text), text

        self.tokens_id = pad_sequence(transposed_data[0], batch_first=True)
        self.length = transposed_data[1]
        self.text = transposed_data[2]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        return self


def collate_fn_predict(batch):
    return Batch_reader_predict(batch)


Selection_loader_predict = partial(DataLoader, collate_fn=collate_fn_predict, pin_memory=True)
