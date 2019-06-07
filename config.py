import os
import json

from lib.preprocessings.chinese_selection import Chinese_selection_preprocessing


class Hyper(object):
    def __init__(self, path: str):
        self.__dict__ = json.load(open(path, 'r'))


class Config(object):
    def __init__(self):
        self.hyper = Hyper('experiments/chinese_selection_re.json')
        self.preprocessor = Chinese_selection_preprocessing(self.hyper)

    def preprocessing(self):
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()

    def run(self):
        print(self.hyper.__dict__)

        self.preprocessing()


if __name__ == "__main__":
    config = Config()
    config.run()
