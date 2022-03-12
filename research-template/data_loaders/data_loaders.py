from torchvision import transforms
from base import BaseDataLoader
from datasets import CaliforniaNDImageDataset

class CaliforniaNDDataLoader(BaseDataLoader):
    """
    California near duplicate image data loader
    """
    def __init__(self, batch_size, data_dir=None, validation_split=0.0, shuffle=False, num_workers=0):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        if data_dir is None:
            self.dataset = CaliforniaNDImageDataset(transform=trsfm)
        else:
            self.dataset = CaliforniaNDImageDataset(data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        