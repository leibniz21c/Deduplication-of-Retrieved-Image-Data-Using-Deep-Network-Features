from torch.nn import Module
from torch import cat
import torch

from dupimage.helper import get_device, get_dataloader, ImageDataset

class BaseExtractor(Module):   
    """
    Must override it 

    model 
    device
    root_dir
    dataset
    dataloader
    labels
    """ 
    def __init__(self, root_dir='./', suffix='jpg', batch_size=64, num_workers=1):
        super().__init__()
        self.device = get_device()
        self.root_dir = root_dir

        # Get Dataset and Dataloader objects from root_dir
        self.dataset = ImageDataset(root_dir=root_dir, suffix=suffix)
        self.dataloader = get_dataloader(self.dataset, batch_size, num_workers)
        
        # Get labels
        self.labels = [label.split('/')[-2] + '/' + label.split('/')[-1] for label in self.dataset.files]
    
    def get_feature_vectors(self, n_components=25088 + 6272):
        """
        
        """
        features = None

        # get features
        for batch in self.dataloader:
            if self.device != 'cpu':
                batch[0] = batch[0].to(self.device).float()
            output = self(batch[0])
            if features == None:
                features = output
            else:
                features = cat([features, output], dim=0)
            del batch
            torch.cuda.empty_cache()

        return features