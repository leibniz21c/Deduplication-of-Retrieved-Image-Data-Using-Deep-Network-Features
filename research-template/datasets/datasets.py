from itertools import combinations
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
    def __init__(self, root=CALIFORNIA_ND_DATASET_PATH, transform=None, preprocessed_path=None):
        self._images_paths = natsorted(glob(root + 'Photos/' + '*.jpg'))
        self._transform = transform
        self._preprocessed = preprocessed_path

        if self._preprocessed is not None:
            self._image_features = np.load(preprocessed_path)
        
            
    def __getitem__(self, index):
        """Get item from california near duplicate image dataset class

        Args:
            index (int): index must be in [0, 701).

        Raises:
            IndexError: If index isn't in [0, 701), raise this error.

        Returns:
            torch.Tensor: image tensor
            str: image index
        """
        if self._preprocessed is not None:
            return self._image_features[index, :]

        if self._transform:
            return self._transform(Image.open(self._images_paths[index])), self._images_paths[index].split('/')[-1]
        return Image.open(self._images_paths[index]), self._images_paths[index].split('/')[-1]
        
    def __len__(self):
        """Get length of dataset

        Returns:
            int: length of dataset
        """
        return len(self._images_paths)

    
class CaliforniaNDPairsDataset(Dataset):
    """Pytorch california near duplicate ground truth dataset class

    """
    def __init__(self, root=CALIFORNIA_ND_DATASET_PATH, corr_threshold=0.5, transform=None, preprocessed_path=None, nnd_approx_equal=False):
        # Create california near duplicate images dataset and nind pairs
        self._images = CaliforniaNDImageDataset(root=root, transform=transform, preprocessed_path=preprocessed_path)
        self._nd_table = (np.load(root + 'Correlation_matrices/gt_all.npy') >= corr_threshold).astype(np.int)

        # Number of non near-duplicate image pairs approximately equal to near-duplicate image pairs
        # (2021, Sensors, Yi Zhang et al.)
        if nnd_approx_equal:
            # near-duplicate pairs
            nd_pairs = np.where(self._nd_table == 1)
            num_nd_pairs = len(nd_pairs[0])
            nd_pairs = [(nd_pairs[0][i], nd_pairs[1][i]) for i in range(num_nd_pairs) if nd_pairs[0][i] < nd_pairs[1][i]]
            num_nd_pairs = len(nd_pairs)

            # non near-duplicate pairs
            i = 0
            nnd_pairs = []
            checked = np.zeros(self._nd_table.shape, dtype=np.bool)
            len_images = len(self._images)
            while i < num_nd_pairs:
                random_pair = np.random.randint(len_images, size=2)
                if random_pair[0] < random_pair[1] and not checked[random_pair[0]][random_pair[1]]:
                    nnd_pairs.append((random_pair[0], random_pair[1]))
                    checked[random_pair[0]][random_pair[1]] = True
                    i += 1
            
            # create pairs
            self._pairs = tuple(nd_pairs + nnd_pairs)
        else:
            # create pairs
            self._pairs = tuple(combinations(range(self._nd_table.shape[0]), 2))


    def __getitem__(self, index):
        return (self._images[self._pairs[index][0]], self._images[self._pairs[index][1]]), self._nd_table[self._pairs[index]]

    
    def __len__(self):
        return len(self._pairs)