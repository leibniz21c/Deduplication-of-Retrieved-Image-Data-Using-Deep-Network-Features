from shutil import copyfile
import os

from dupimage.extractor.vgg19_custom_layer.features6272_features25088 import Features6272_Features25088
from dupimage.tools import get_clusters

DEFUALT_RESULT_DIR_PATH = '/result/'

def remove_dup_images(
    root_dir: str,
    suffix='jpg',
    result_dir=None
):
    # default result directory
    if result_dir == None:
        if not os.path.exists(root_dir + DEFUALT_RESULT_DIR_PATH):
            os.makedirs(root_dir + DEFUALT_RESULT_DIR_PATH)
        result_dir = root_dir + DEFUALT_RESULT_DIR_PATH

    # get cluster
    predict, labels = get_clusters(
        extractor=Features6272_Features25088(
            root_dir=root_dir, 
            suffix=suffix, 
            batch_size=64, 
            num_workers=1),
        threshold=0.2)
    
    # lables
    labels_set = set(labels)
    predicted_labels_set = set()

    n_clusters = max(predict)
    cluster_list = [False for i in range(n_clusters + 1)]

    for i, cluster in enumerate(predict):
        if cluster_list[cluster] == False:
            cluster_list[cluster] = True    
            predicted_labels_set.add(labels[i])

    for image in predicted_labels_set:
        copyfile(image, result_dir + '/' + image.split('/')[-1])