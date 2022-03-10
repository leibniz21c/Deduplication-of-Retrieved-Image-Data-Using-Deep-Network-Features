from itertools import product
import numpy as np
import torch.nn as F
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from torch import cat
from torchvision import models
from base import BaseModel

class DupimageModel(BaseModel):
    """DupimageModel from KSC 2021, 
    https://github.com/ndo04343/dupimage/blob/main/dupimage/extractor/vgg19_custom_layer/features6272_features25088.py
    """
    def __init__(self):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True)

        self.feature_layer1 = F.Sequential(
            vgg19.features[:10],
            vgg19.avgpool,
            F.Flatten()
        )
        self.feature_layer2 = F.Sequential(
            vgg19.features,
            vgg19.avgpool,
            F.Flatten()
        )

    def __str__(self):
        return """Model : DupimageModel near-duplicate image retrieval from KSC 2021"""

    def forward(self, x):
        output_features1 = self.feature_layer1(x)
        output_features2 = self.feature_layer2(x)
        return cat([output_features1, output_features2], dim=1)

    @classmethod
    def pca_embedding(cls, features, n_components):
        features = features.cpu().numpy()
        pca = PCA(n_components=n_components)
        embedded_features = pca.fit_transform(X=features)

        return embedded_features, pca.explained_variance_ratio_

    @classmethod
    def agglomerative_clustering(cls, features, threshold=0.2):
        cluster = AgglomerativeClustering(
            n_clusters=None,
            affinity='cosine',
            linkage='single',
            distance_threshold=threshold
        )
        predict_cluster = cluster.fit_predict(features)
        num_labels = len(predict_cluster)
        num_cluster = cluster.n_clusters_

        return predict_cluster, num_labels, num_cluster