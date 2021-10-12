#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def cosine_dist(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

def euclidean_dist(x, y):
    return np.linalg.norm(x - y)
