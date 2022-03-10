from torchvision import models
from torch import cat
import torch.nn as F

from dupimage.extractor.base_extractor import BaseExtractor

class Features6272_Features25088(BaseExtractor):
    """
    
    """
    def __init__(self,root_dir='./', suffix='jpg', batch_size=64, num_workers=1):
        super().__init__(root_dir=root_dir, suffix=suffix, batch_size=batch_size, num_workers=num_workers)
    
        # base model
        vgg19 = models.vgg19(pretrained=True)

        # get model : features 6272
        self.features6272 = F.Sequential(
            vgg19.features[:10],
            vgg19.avgpool,
            F.Flatten()
        ).to(self.device)

        # get model : features 25088
        self.features25088 = F.Sequential(
            vgg19.features,
            vgg19.avgpool,
            F.Flatten()
        ).to(self.device)
        
    def forward(self, x):
        """
        
        """
        output_features6272 = self.features6272(x)
        output_features25088 = self.features25088(x)
    
        return cat([output_features6272, output_features25088], dim=1)