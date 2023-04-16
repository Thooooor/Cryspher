import torch
from pymatgen.core import Structure
from torch_geometric.data import Data, InMemoryDataset, makedirs
import os
import json
from typing import List
from tqdm import tqdm
from sklearn.model_selection import train_test_split

FULL_DATASETS = ["band_gap", "e_form", "perovskites", "log_gvrh", "log_kvrh", "dielectric", "jdft2d", "phonons"]
SUB_DATASETS = ["band_gap", "e_form", "perovskites", "log_gvrh", "log_kvrh"]


class CrystalBase(InMemoryDataset):
    def __init__(
        self, 
        dataset, 
        data_dir, 
        data_file, 
        split="train", 
        train_ratio=0.6, 
        valid_ratio=0.2, 
        test_ratio=0.2, 
        transform=None, 
        pre_transform=None, 
        pre_filter=None, 
        load_subset=False, 
        subset_ratio=0.2
        ):
        self.data_file = data_file
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.subset = True if dataset in SUB_DATASETS else False
        self.subset_ratio = subset_ratio
        assert split in ['train', 'valid', 'test']
        super().__init__(data_dir, transform, pre_transform, pre_filter)
        if load_subset is True:
            path = os.path.join(self.subset_dir, split + '.pt')
        else:
            path = os.path.join(self.processed_dir, split + '.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_dir(self) -> str:
        return self.root
    
    @property
    def raw_file_names(self) -> List[str]:
        return [self.data_file]
    
    @property
    def processed_file_names(self) -> List[str]:
        return ['train.pt', 'valid.pt', 'test.pt']
    
    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")
    
    @property
    def subset_dir(self) -> str:
        return os.path.join(self.root, "subset")
    
    def process(self):
        init_path = os.path.join(self.root, "raw", "atom_init.json")
        with open(init_path, 'r') as f:
            atom_dict = json.load(f)
        file_path = os.path.join(self.root, "raw", self.data_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
        index = data['index']
        raw_data = data['data']

        data_list = []
        for idx in tqdm(index):
            structure_dict, target = raw_data[idx]
            structure = Structure.from_dict(structure_dict)
            data_list.append(from_structure_to_graph(structure, target, atom_dict))
            
        train_index, valid_test_index = train_test_split(index, train_size=self.train_ratio)
        valid_index, test_index = train_test_split(valid_test_index, train_size=self.valid_ratio/(self.valid_ratio+self.test_ratio))
        
        train_data = [data_list[i] for i in train_index]
        valid_data = [data_list[i] for i in valid_index]
        test_data = [data_list[i] for i in test_index]
        
        torch.save(self.collate(train_data), os.path.join(self.processed_dir, 'train.pt'))
        torch.save(self.collate(valid_data), os.path.join(self.processed_dir, 'valid.pt'))
        torch.save(self.collate(test_data), os.path.join(self.processed_dir, 'test.pt'))

        if self.subset is True:
            makedirs(self.subset_dir)
            sub_index = index[:int(len(index)*self.subset_ratio)]
            sub_train_index, sub_valid_test_index = train_test_split(sub_index, train_size=self.train_ratio)
            sub_valid_index, sub_test_index = train_test_split(sub_valid_test_index, train_size=self.valid_ratio/(self.valid_ratio+self.test_ratio))
            
            sub_train_data = [data_list[i] for i in sub_train_index]
            sub_valid_data = [data_list[i] for i in sub_valid_index]
            sub_test_data = [data_list[i] for i in sub_test_index]
            
            torch.save(self.collate(sub_train_data), os.path.join(self.subset_dir, 'train.pt'))
            torch.save(self.collate(sub_valid_data), os.path.join(self.subset_dir, 'valid.pt'))
            torch.save(self.collate(sub_test_data), os.path.join(self.subset_dir, 'test.pt'))


def from_structure_to_graph(structure, target, atom_dict, max_neigh=12, radius=8):
    """Convert pymatgen.core.structure.Structure to torch_geometric.data.Data

    Args:
        structure (_type_): _description_
        target (_type_): _description_
    """
    atomic_numbers = structure.atomic_numbers
    node_features = [atom_dict[str(atomic_number)] for atomic_number in atomic_numbers]
    # node_features = [atomic_number for atomic_number in atomic_numbers]
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    all_neighbors = structure.get_all_neighbors(radius)
    while len(all_neighbors) == 0:
        radius += 1
        all_neighbors = structure.get_all_neighbors(radius)
    edge_index = []
    edge_attr = []
    for i, neighbors in enumerate(all_neighbors):
        count = 0
        while len(neighbors) < max_neigh:
            radius += 1
            neighbors = structure.get_all_neighbors(radius)[i]
        for neighbor in neighbors:
            edge_index.append([i, neighbor.index])
            edge_attr.append([neighbor[1]])
            count += 1
            if count >= max_neigh:
                break
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    
    return Data(x=node_features, edge_index=edge_index, y=torch.tensor([target]), edge_attr=edge_attr)
    # return node_features, edge_index, edge_attr, target
