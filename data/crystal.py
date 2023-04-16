from torch_geometric.loader import DataLoader
from .base import CrystalBase


class CrystalDataset:
    def __init__(
        self, 
        dataset, 
        data_dir, 
        data_file, 
        train_ratio=0.6, 
        valid_ratio=0.2, 
        test_ratio=0.2, 
        batch_size=1024, 
        load_subset=False, 
        ):
        self.batch_size = batch_size
        self.train_data = CrystalBase(dataset, data_dir, data_file, split="train", train_ratio=train_ratio, 
                                      valid_ratio=valid_ratio, test_ratio=test_ratio, load_subset=load_subset)
        self.valid_data = CrystalBase(dataset, data_dir, data_file, split="valid", train_ratio=train_ratio, 
                                      valid_ratio=valid_ratio, test_ratio=test_ratio, load_subset=load_subset)
        self.test_data = CrystalBase(dataset, data_dir, data_file, split="test", train_ratio=train_ratio, 
                                     valid_ratio=valid_ratio, test_ratio=test_ratio, load_subset=load_subset)

    @property
    def train_loader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
    
    @property
    def valid_loader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size)
    
    @property
    def test_loader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
