from torch import cat
import torch.nn as F
from torchvision import models
from base import BaseModel

class VGGPartialCompose(BaseModel):
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True)

        self.feature_layer1 = F.Sequential(
            vgg19.features[:10], vgg19.avgpool, F.Flatten()
        )
        self.feature_layer2 = F.Sequential(vgg19.features, vgg19.avgpool, F.Flatten())

    def __str__(self):
        return """VGG19PartialCompose()"""

    def forward(self, x):
        return cat([self.feature_layer1(x), self.feature_layer2(x)], dim=1)