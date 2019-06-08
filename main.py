import os
import json

from lib.preprocessings import Chinese_selection_preprocessing
from lib.config import Hyper


class Runner(object):
    def __init__(self):
        self.hyper = Hyper('experiments/chinese_selection_re.json')
        self.preprocessor = Chinese_selection_preprocessing(self.hyper)

    def preprocessing(self):
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        self.preprocessor.gen_bio_vocab()

    def run(self):
        print(self.hyper.__dict__)

        # self.preprocessing()


if __name__ == "__main__":
    config = Runner()
    config.run()
