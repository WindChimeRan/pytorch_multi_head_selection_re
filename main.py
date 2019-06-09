import os
import json
import time

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from lib.preprocessings import Chinese_selection_preprocessing
from lib.dataloaders import Selection_Dataset, Selection_loader
from lib.models import MultiHeadSelection
from lib.config import Hyper

from torch.optim import Adam, SGD


class Runner(object):
    def __init__(self):
        self.hyper = Hyper('experiments/chinese_selection_re.json')
        self.gpu = self.hyper.gpu
        self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        self.model = MultiHeadSelection(self.hyper).cuda(self.gpu)
        # self.model = MultiHeadSelection(self.hyper)
        self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
        

    def _optimizer(self, name, model):
        m = {'adam': Adam(model.parameters()), 'sgd': SGD(model.parameters(), lr=0.5)}
        return m[name]

    def preprocessing(self):
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self):
        # print(self.hyper.__dict__)
        # self.preprocessing()
        self.train()

    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=100, pin_memory=True)
        for epoch in range(self.hyper.epoch_num):

            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_ndx, sample in pbar:
                self.model.train()

                self.optimizer.zero_grad()
                output = self.model(sample, inference=False)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description("loss: {:.2f}, epoch: {}/{}:".format(
                    loss.item(), epoch, self.hyper.epoch_num))


if __name__ == "__main__":
    config = Runner()
    config.run()
