import numpy as np
from PIL import Image, ImageFile
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset

# Set truncation config
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
CALIFORNIA_ND_DATASET_PATH = '/datasets/california-nd/'

class CaliforniaNDImageDataset(Dataset):
    """Pytorch California near duplicate image dataset class

    """
    def __init__(self, root=CALIFORNIA_ND_DATASET_PATH, transform=None):
        self._images_paths = natsorted(glob(root + 'Photos/' + '*.jpg'))
        self._transform = transform
            
    def __getitem__(self, index):
        """Get item from california near duplicate image dataset class

        Args:
            index (int): index must be in $ 0 \le index < 701 $

        Raises:
            IndexError: If index isn't in [0, 701), raise this error.

        Returns:
            torch.Tensor: image tensor
            str: image index
        """
        if self._transform:
            return self._transform(Image.open(self._images_paths[index])), self._images_paths[index].split('/')[-1]
        return Image.open(self._images_paths[index]), self._images_paths[index].split('/')[-1]
        
    def __len__(self):
        """Get length of dataset

        Returns:
            int: length of dataset
        """
        return len(self._images_paths)
    
    def get_labels(self):
        return (label.split('/')[-1] for label in self._images_paths)
    
class CaliforniaNDTruthNetwork:
    """Pytorch california near duplicate ground truth dataset class

    """
    def __init__(self, root=CALIFORNIA_ND_DATASET_PATH, threshold=0.5, target=True):
        # Create empty cluster matrix
        self._data = np.load(root + 'Correlation_matrices/gt_all.npy')
        
        # Target zero adjacency matrix
        if target == False:
            self._data = np.zeros(self._data.shape)
            return
        
        # Thresholding
        self._data[self._data < threshold] = 0
        self._data[self._data >= threshold] = 1
        
    def add_edge(self, u, v):
        self._data[u, v] = 1
            
    def get_adj_matrix(self):
        return self._data