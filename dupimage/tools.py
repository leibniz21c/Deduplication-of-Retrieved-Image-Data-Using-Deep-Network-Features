#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from dupimage.extractor.base_extractor import BaseExtractor
from dupimage import dist_metric

def get_distmat(
    features,
    distance_metric=dist_metric.cosine_dist):
    """
    
    """
    # get number of data
    num_of_data = features.shape

    # create distance metrix
    dist_mat = np.empty(num_of_data)

    # main logic
    for i, ith_feature in enumerate(features):
        for j, jth_feature in enumerate(features):
            if i > j: dist_mat[i, j] = dist_mat[j, i] = distance_metric(ith_feature, jth_feature)
            elif i == j: dist_mat[i, j] = 0
    
    return dist_mat

def get_clusters(
    extractor: BaseExtractor,
    threshold=0.2):
    """
    Hierarchical Clustering(Dendrogram) Based Cluster Generator
        pass
    """
    # Get labels and features
    with torch.no_grad():
        labels = extractor.labels
        features = extractor.get_feature_vectors().cpu().numpy()
        torch.cuda.empty_cache()

    # embedding by pca
    pca = PCA(n_components=len(extractor.labels))
    embedded_features = pca.fit_transform(X=features)
    
    # clustering
    cluster = AgglomerativeClustering(
        n_clusters=None,
        affinity='cosine',
        linkage='single',
        distance_threshold=threshold)
    
    predict = cluster.fit_predict(embedded_features)

    return predict, labels
    