import os
import json

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset


class Selection_loader(Dataset):
    def __init__(self, hyper, dataset):
        self.data_root = hyper.data_root

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

        for line in open(os.path.join(self.data_root, dataset), 'r'):
            line = line.strip("\n")
            instance = json.loads(line)

            self.selection_list.append(instance['selection'])
            self.text_list.append(instance['text'])
            self.bio_list.append(instance['bio'])
            self.spo_list.append(instance['spo_list'])

    def __getitem__(self, index):
        selection = self.selection_list[index]
        text = self.text_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]

        tokens = self.text2tensor(text)
        bio = self.bio2tensor(bio)
        selection = self.selection2tensor(selection)

        return text, bio, selection, spo

    def __len__(self):
        return len(self.text_list)

    def text2tensor(self):
        pass

    def bio2tensor(self):
        pass

    def selection2tensor(self):
        pass


class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


def collate_fn(batch):
    return Batch_reader(batch)
