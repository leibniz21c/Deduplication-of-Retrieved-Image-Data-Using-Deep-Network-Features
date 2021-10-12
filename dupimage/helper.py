#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset
from glob import iglob
from PIL import Image

def get_device():
    """
    
    """
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_dataloader(dataset: Dataset, batch_size=64, num_worker=1) -> DataLoader:
    """
    
    """
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_worker)

class ImageDataset(Dataset):
    """
    
    """
    def __init__(self, root_dir, suffix, size=(224, 224)):
        # Get files
        if root_dir[-1] == '/':
            self.files = list(iglob(root_dir + '**/*.' + suffix, recursive=True))
        else:
            self.files = list(iglob(root_dir + '/**/*.' + suffix, recursive=True))
        self.files.sort()
        
        # Get size
        self.size = size
    
    def __len__(self):
        ''' number of files '''
        return len(self.files)

    def __getitem__(self, idx):
        """
        open with RGB, resize, and normalization
        """
        return torch.from_numpy(np.array(Image.open(self.files[idx]).convert("RGB").resize(self.size), dtype=np.float64)/255.0).reshape(3, 224, 224), self.files[idx]