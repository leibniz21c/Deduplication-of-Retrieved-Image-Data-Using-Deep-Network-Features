import os
import json
from itertools import combinations
import numpy as np
import torch
from scipy import sparse
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from datasets import ImageDataset


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def mfnd_all_sampling(root, sample_name):
    # Create sample dataset directory
    if not os.path.exists(root + sample_name):
        os.makedirs(root + sample_name)

    if not os.path.exists(root + sample_name + "/images"):
        os.makedirs(root + sample_name + "/images")

    # Create empty near duplicate sparse matrix with (1000000, 1000000) shape
    nd_matrix = sparse.lil_matrix((1000000, 1000000), dtype=np.int8)

    # Duplicate pair : 1
    with open(root + "mfnd/" + "duplicates.txt") as f:
        lines = f.readlines()

        for line in lines:
            for i, j in combinations(line.strip().split(" "), 2):
                if i < j:
                    nd_matrix[int(i), int(j)] = 1
                else:
                    nd_matrix[int(j), int(i)] = 1

    # IND pair : 1
    with open(root + "mfnd/" + "IND_clusters.txt") as f:
        lines = f.readlines()

        for line in lines:
            for i, j in combinations(line.strip().split(" "), 2):
                if i < j:
                    nd_matrix[int(i), int(j)] = 1
                else:
                    nd_matrix[int(j), int(i)] = 1

    # NIND pair : 1
    with open(root + "mfnd/" + "NIND_clusters.txt") as f:
        lines = f.readlines()

        for line in lines:
            for i, j in combinations(line.strip().split(" "), 2):
                if i < j:
                    nd_matrix[int(i), int(j)] = 1
                else:
                    nd_matrix[int(j), int(i)] = 1

    # Number of non near-duplicate image pairs approximately equal to near-duplicate image pairs
    # (2021, Sensors, Yi Zhang et al.)
    # near-duplicate pairs
    nd_pairs = nd_matrix.nonzero()
    nd_pairs = [(nd_pairs[0][i], nd_pairs[1][i]) for i in range(len(nd_pairs[0]))]
    num_nd_pairs = len(nd_pairs)

    # non near-duplicate pairs
    i = 0
    nnd_pairs = []
    checked = sparse.lil_matrix(nd_matrix.shape, dtype=np.bool)

    len_images = 1000000
    while i < num_nd_pairs:
        random_pair = np.random.randint(len_images, size=2)
        if random_pair[0] < random_pair[1]:
            if not checked[random_pair[0], random_pair[1]]:
                nnd_pairs.append((random_pair[0], random_pair[1]))
                checked[random_pair[0], random_pair[1]] = True
                i += 1
        else:
            if not checked[random_pair[1], random_pair[0]]:
                nnd_pairs.append((random_pair[1], random_pair[0]))
                checked[random_pair[1], random_pair[0]] = True
                i += 1

    # save nd pairs and sample images
    mirflickrs = MirFlickr1MDataset(root=root)
    checked = np.zeros(1000000, dtype=np.bool)
    with open(root + sample_name + "/nd_pairs.txt", "w") as f:
        for i, j in tqdm(nd_pairs, desc="Save ND pairs"):
            f.write("{} {}\n".format(i, j))
            if not checked[i]:
                mirflickrs[i][0].save(root + sample_name + "/images/" + str(i) + ".jpg")
                checked[i] = True
            if not checked[j]:
                mirflickrs[j][0].save(root + sample_name + "/images/" + str(j) + ".jpg")
                checked[j] = True

    # save nnd pairs
    checked = np.zeros(1000000, dtype=np.bool)
    with open(root + sample_name + "/nnd_pairs.txt", "w") as f:
        for i, j in tqdm(nnd_pairs, desc="Save NND pairs"):
            f.write("{} {}\n".format(i, j))
            if not checked[i]:
                mirflickrs[i][0].save(root + sample_name + "/images/" + str(i) + ".jpg")
                checked[i] = True
            if not checked[j]:
                mirflickrs[j][0].save(root + sample_name + "/images/" + str(j) + ".jpg")
                checked[j] = True


def mfnd_ind_sampling(root, sample_name):
    # Create sample dataset directory
    if not os.path.exists(root + sample_name):
        os.makedirs(root + sample_name)

    if not os.path.exists(root + sample_name + "/images"):
        os.makedirs(root + sample_name + "/images")

    # Create empty near duplicate sparse matrix with (1000000, 1000000) shape
    nd_matrix = sparse.lil_matrix((1000000, 1000000), dtype=np.int8)

    # Duplicate pair : 1
    with open(root + "mfnd/" + "duplicates.txt") as f:
        lines = f.readlines()

        for line in lines:
            for i, j in combinations(line.strip().split(" "), 2):
                if i < j:
                    nd_matrix[int(i), int(j)] = 1
                else:
                    nd_matrix[int(j), int(i)] = 1

    # IND pair : 1
    with open(root + "mfnd/" + "IND_clusters.txt") as f:
        lines = f.readlines()

        for line in lines:
            for i, j in combinations(line.strip().split(" "), 2):
                if i < j:
                    nd_matrix[int(i), int(j)] = 1
                else:
                    nd_matrix[int(j), int(i)] = 1


    # Number of non near-duplicate image pairs approximately equal to near-duplicate image pairs
    # (2021, Sensors, Yi Zhang et al.)
    # near-duplicate pairs
    nd_pairs = nd_matrix.nonzero()
    nd_pairs = [(nd_pairs[0][i], nd_pairs[1][i]) for i in range(len(nd_pairs[0]))]
    num_nd_pairs = len(nd_pairs)

    # non near-duplicate pairs
    i = 0
    nnd_pairs = []
    checked = sparse.lil_matrix(nd_matrix.shape, dtype=np.bool)

    len_images = 1000000
    while i < num_nd_pairs:
        random_pair = np.random.randint(len_images, size=2)
        if random_pair[0] < random_pair[1]:
            if not checked[random_pair[0], random_pair[1]]:
                nnd_pairs.append((random_pair[0], random_pair[1]))
                checked[random_pair[0], random_pair[1]] = True
                i += 1
        else:
            if not checked[random_pair[1], random_pair[0]]:
                nnd_pairs.append((random_pair[1], random_pair[0]))
                checked[random_pair[1], random_pair[0]] = True
                i += 1

    # save nd pairs and sample images
    mirflickrs = MirFlickr1MDataset(root=root)
    checked = np.zeros(1000000, dtype=np.bool)
    with open(root + sample_name + "/nd_pairs.txt", "w") as f:
        for i, j in tqdm(nd_pairs, desc="Save ND pairs"):
            f.write("{} {}\n".format(i, j))
            if not checked[i]:
                mirflickrs[i][0].save(root + sample_name + "/images/" + str(i) + ".jpg")
                checked[i] = True
            if not checked[j]:
                mirflickrs[j][0].save(root + sample_name + "/images/" + str(j) + ".jpg")
                checked[j] = True

    # save nnd pairs
    checked = np.zeros(1000000, dtype=np.bool)
    with open(root + sample_name + "/nnd_pairs.txt", "w") as f:
        for i, j in tqdm(nnd_pairs, desc="Save NND pairs"):
            f.write("{} {}\n".format(i, j))
            if not checked[i]:
                mirflickrs[i][0].save(root + sample_name + "/images/" + str(i) + ".jpg")
                checked[i] = True
            if not checked[j]:
                mirflickrs[j][0].save(root + sample_name + "/images/" + str(j) + ".jpg")
                checked[j] = True


def california_nd_sampling(root, sample_name):
    # Create sample dataset directory
    if not os.path.exists(root + sample_name):
        os.makedirs(root + sample_name)

    if not os.path.exists(root + sample_name + "/images"):
        os.makedirs(root + sample_name + "/images")

    # Create empty near duplicate sparse matrix with (1000000, 1000000) shape
    nd_matrix = sparse.lil_matrix((1000000, 1000000), dtype=np.int8)

    # Duplicate pair : 1
    with open(root + "mfnd/" + "duplicates.txt") as f:
        lines = f.readlines()

        for line in lines:
            for i, j in combinations(line.strip().split(" "), 2):
                if i < j:
                    nd_matrix[int(i), int(j)] = 1
                else:
                    nd_matrix[int(j), int(i)] = 1

    # IND pair : 1
    with open(root + "mfnd/" + "IND_clusters.txt") as f:
        lines = f.readlines()

        for line in lines:
            for i, j in combinations(line.strip().split(" "), 2):
                if i < j:
                    nd_matrix[int(i), int(j)] = 1
                else:
                    nd_matrix[int(j), int(i)] = 1


    # Number of non near-duplicate image pairs approximately equal to near-duplicate image pairs
    # (2021, Sensors, Yi Zhang et al.)
    # near-duplicate pairs
    nd_pairs = nd_matrix.nonzero()
    nd_pairs = [(nd_pairs[0][i], nd_pairs[1][i]) for i in range(len(nd_pairs[0]))]
    num_nd_pairs = len(nd_pairs)

    # non near-duplicate pairs
    i = 0
    nnd_pairs = []
    checked = sparse.lil_matrix(nd_matrix.shape, dtype=np.bool)

    len_images = 1000000
    while i < num_nd_pairs:
        random_pair = np.random.randint(len_images, size=2)
        if random_pair[0] < random_pair[1]:
            if not checked[random_pair[0], random_pair[1]]:
                nnd_pairs.append((random_pair[0], random_pair[1]))
                checked[random_pair[0], random_pair[1]] = True
                i += 1
        else:
            if not checked[random_pair[1], random_pair[0]]:
                nnd_pairs.append((random_pair[1], random_pair[0]))
                checked[random_pair[1], random_pair[0]] = True
                i += 1

    # save nd pairs and sample images
    mirflickrs = MirFlickr1MDataset(root=root)
    checked = np.zeros(1000000, dtype=np.bool)
    with open(root + sample_name + "/nd_pairs.txt", "w") as f:
        for i, j in tqdm(nd_pairs, desc="Save ND pairs"):
            f.write("{} {}\n".format(i, j))
            if not checked[i]:
                mirflickrs[i][0].save(root + sample_name + "/images/" + str(i) + ".jpg")
                checked[i] = True
            if not checked[j]:
                mirflickrs[j][0].save(root + sample_name + "/images/" + str(j) + ".jpg")
                checked[j] = True

    # save nnd pairs
    checked = np.zeros(1000000, dtype=np.bool)
    with open(root + sample_name + "/nnd_pairs.txt", "w") as f:
        for i, j in tqdm(nnd_pairs, desc="Save NND pairs"):
            f.write("{} {}\n".format(i, j))
            if not checked[i]:
                mirflickrs[i][0].save(root + sample_name + "/images/" + str(i) + ".jpg")
                checked[i] = True
            if not checked[j]:
                mirflickrs[j][0].save(root + sample_name + "/images/" + str(j) + ".jpg")
                checked[j] = True