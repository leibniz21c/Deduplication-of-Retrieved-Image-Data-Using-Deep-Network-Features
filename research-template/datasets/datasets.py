from glob import glob
from natsort import natsorted
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

# Set truncation config
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MirFlickr1MDataset(Dataset):
    def __init__(self, root, transform=None):
        self.__images_paths = root + "images/"
        self.__transform = transform

    def __getitem__(self, index):
        if index < 0 or index >= 1000000:
            raise IndexError("Should be in 0 < index <= 1000000")
        path = self.__images_paths + str(index // 10000) + "/" + (str(index) + ".jpg")

        if self.__transform:
            return self.__transform(Image.open(path)), str(index)
        return Image.open(path), str(index)

    def __len__(self):
        return len(self.__images_paths)

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        # Get images path
        self.__base_path = root
        self.__images_paths = natsorted(glob(root + "*.jpg"))
        self.__images_labels = [path.split("/")[-1] for path in self.__images_paths]
        self.__transform = False if transform is None else transform

    def __getitem__(self, key):
        # index and label check
        index = self.__images_paths.index(self.__base_path + key) if type(key) is str else key

        if self.__transform:
            return (
                self.__transform(Image.open(self.__images_paths[index])),
                self.__images_labels[index],
            )
        return (
            Image.open(self.__images_paths[index]),
            self.__images_labels[index],
        )

    def __len__(self):
        return len(self.__images_paths)

class ProcessedImageDataset(Dataset):
    def __init__(self, root, prep_name):
        
        # Get images path
        self.__base_path = root
        self.__images_labels = [path.split("/")[-1] for path in natsorted(glob(root + "images/*.jpg"))]
        self.__dataset = torch.load(self.__base_path + "preprocessed/" + prep_name + ".pt")

    def __getitem__(self, key):
        # index and label check
        index = self.__images_labels.index(key) if type(key) is str else key
        return (
            self.__dataset[index, :],
            self.__images_labels[index],
        )

    def __len__(self):
        return len(self.__images_labels)

class ProcessedPairDataset(Dataset):
    def __init__(self, root, prep_name):
        # Processed images dataset
        self.__processed_images_dataset = ProcessedImageDataset(root, prep_name)

        self.__pairs = []
        with open(root + "nd_pairs.txt") as f:
            lines = f.readlines()
            for line in lines:
                i, j = line.strip().split(' ')
                self.__pairs.append(((i + '.jpg', j + '.jpg'), 1))
        
        with open(root + "nnd_pairs.txt") as f:
            lines = f.readlines()
            for line in lines:
                i, j = line.strip().split(' ')
                self.__pairs.append(((i + '.jpg', j + '.jpg'), 0))

    def __getitem__(self, index):
        return (
            self.__processed_images_dataset[self.__pairs[index][0][0]],
            self.__processed_images_dataset[self.__pairs[index][0][1]],
        ), self.__pairs[index][1]

    def __len__(self):
        return len(self.__pairs)