import json

from dataclasses import dataclass

@dataclass
class Hyper(object):
    def __init__(self, path: str):
        self.dataset: str
        self.model: str
        self.data_root: str
        self.raw_data_root: str
        self.train: str
        self.dev: str
        self.relation_vocab: str
        self.print_epoch: int
        self.evaluation_epoch: int
        self.max_text_len: int
        self.cell_name: str
        self.emb_size: int
        self.rel_emb_size: int
        self.hidden_size: int
        self.threshold: float
        self.activation: str
        self.optimizer: str
        self.epoch_num: int
        self.gpu: int

        self.__dict__ = json.load(open(path, 'r'))

    def __post_init__(self):
        pass
