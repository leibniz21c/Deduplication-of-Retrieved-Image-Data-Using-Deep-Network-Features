from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import CaliforniaNDImageDataset, CaliforniaNDPairsDataset

class CaliforniaNDPairDataLoader(DataLoader):
    """
    California near duplicate image pair data loader
    """
    def __init__(self, batch_size=1, data_dir=None, corr_threshold=0.5, num_workers=0, preprocessed_path=False, nnd_approx_equal=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        if data_dir is None:
            self._dataset = CaliforniaNDPairsDataset(
                corr_threshold=corr_threshold, 
                transform=trsfm, 
                preprocessed_path=preprocessed_path, 
                nnd_approx_equal=nnd_approx_equal)
        else:
            self._dataset = CaliforniaNDPairsDataset(
                root=data_dir, 
                corr_threshold=corr_threshold, 
                transform=trsfm, 
                preprocessed_path=preprocessed_path,
                nnd_approx_equal=nnd_approx_equal)
            
        super().__init__(self._dataset, batch_size, num_workers=num_workers)

class CaliforniaNDImageDataLoader(DataLoader):
    """
    California near duplicate image data loader
    """
    def __init__(self, batch_size=1, data_dir=None, shuffle=False, num_workers=0, preprocessed_path=False):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), 
            interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        if data_dir is None:
            self._dataset = CaliforniaNDImageDataset(
                transform=trsfm, 
                preprocessed_path=preprocessed_path)
        else:
            self._dataset = CaliforniaNDImageDataset(
                root=data_dir, 
                transform=trsfm, 
                preprocessed_path=preprocessed_path)
            
        super().__init__(self._dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
