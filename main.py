import os
import json
import time

import torch

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam, SGD

from lib.preprocessings import Chinese_selection_preprocessing
from lib.dataloaders import Selection_Dataset, Selection_loader
from lib.metrics import F1_triplet
from lib.models import MultiHeadSelection
from lib.config import Hyper


class Runner(object):
    def __init__(self):
        self.exp_name = 'chinese_selection_re'
        self.model_dir = 'saved_models'

        # self.hyper = Hyper('experiments/chinese_selection_re.json')
        self.hyper = Hyper(os.path.join('experiments', self.exp_name + '.json')

        self.gpu = self.hyper.gpu
        self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        self.model = MultiHeadSelection(self.hyper).cuda(self.gpu)
        # self.model = MultiHeadSelection(self.hyper)
        self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
        self.metrics = F1_triplet()

    def _optimizer(self, name, model):
        m = {
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.5)
        }
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
        # self.evaluation()

    def load_model(self, epoch):
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, self.exp_name + '_' + epoch)))

    def save_model(self, epoch):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.exp_name + '_' + epoch))

    def evaluation(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=40, pin_memory=True)
        self.metrics.reset()
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                self.metrics(output['selection_triplets'], output['spo_gold'])

            result = self.metrics.get_metric()
            print(', '.join([
                "%s: %.4f" % (name, value)
                for name, value in result.items() if not name.startswith("_")
            ]) + " ||")

    def train(self):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=100, pin_memory=True)

        for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:

                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.save_model(epoch)

            if epoch % self.hyper.print_epoch == 0 and epoch != 0:
                self.evaluation()


if __name__ == "__main__":
    config = Runner()
    config.run()
