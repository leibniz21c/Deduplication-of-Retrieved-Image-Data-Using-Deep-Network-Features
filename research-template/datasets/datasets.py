from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset

# Set truncation config
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
OXFORD_DATASET_PATH = '/datasets/oxford5k/'
PARIS_DATASET_PATH = '/datasets/paris6k/'
HOLIDAYS_DATASET_PATH = '/datasets/holidays/'

class OxfordImageDataset(Dataset):
    """Pytorch oxford 5k image dataset class

    """
    def __init__(self, root=OXFORD_DATASET_PATH, transform=None):
        self._images_paths = glob(root + 'images/' + '*')
        self.transform = transform
            
    def __getitem__(self, index):
        """Get item from oxford 5K image dataset class

        Args:
            index (int): index must be in $ 0 \le index < 5062 $

        Raises:
            IndexError: If index isn't in [0, 5063), raise this error.

        Returns:
            torch.Tensor: image tensor
            str: image index
        """
        if self.transform:
            return self.transform(Image.open(self._images_paths[index])), self._images_paths[index].split('/')[-1]
        return Image.open(self._images_paths[index]), self._images_paths[index].split('/')[-1]
        
    def __len__(self):
        """Get length of dataset

        Returns:
            int: length of dataset
        """
        return len(self._images_paths)
    
    def get_labels(self):
        return (label.split('/')[-1] for label in self._images_paths)
        
    
class OxfordGroundTruthNetwork:
    """Pytorch oxford 5k ground truth dataset class

    """
    def __init__(self, target=True, root=OXFORD_DATASET_PATH):   # 'ground_truth/'
        # Create empty cluster matrix
        dataset = OxfordImageDataset(root)
        labels = [item.split('.')[0] for item in dataset.get_labels()]
        self._adjacency_list = nx.Graph()
        [self._adjacency_list.add_node(label) for label in labels]
        
        # Insert target values
        if target:
            dataset_paths = natsorted(glob(root + 'ground_truth/' + '*good*' ))
        
            # For each dataset file
            for dataset_path in dataset_paths:
                with open(dataset_path) as f:
                    lines = f.readlines()
                    lines = [line.rstrip() for line in lines]
                    
                    for i, j in product(lines, lines):
                        self._adjacency_list.add_edge(i, j)
        
        # Self duplicate
        for i in labels:
            self._adjacency_list.add_edge(i, i)
            
    def add_edge(self, u, v):
        return self._adjacency_list.add_edge(u, v)
            
    def get_adj_matrix(self):
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)
        adj_mat = nx.adjacency_matrix(self._adjacency_list, dtype=np.int8)
        return adj_mat