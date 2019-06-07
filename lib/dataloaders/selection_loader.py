from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

# https://pytorch.org/docs/stable/data.html
class selection_loader(Dataset):
    def __init__(self, hyper):
        self.dataset = hyper.dataset

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError