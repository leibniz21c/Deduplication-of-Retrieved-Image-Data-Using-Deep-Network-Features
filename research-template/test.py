import argparse
from itertools import product
from time import time
from datetime import timedelta
from operator import itemgetter
from tqdm import tqdm
import torch
import data_loaders.data_loaders as module_data_loader
import datasets.datasets as module_dataset
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

def main(config):
    logger = config.get_logger('test')

    # Time logging
    start_time = time()

    # setup image data_loader instances
    image_data_loader = getattr(module_data_loader, config['image_data_loader']['type'])(
        batch_size=config['image_data_loader']['args']['batch_size'],
        data_dir=config['image_data_loader']['args']['data_dir'],
        shuffle=False,
        validation_split=0.0,
        num_workers=config['n_worker'],
    )

    # setup near duplicate network instances
    target_near_duplicate_network = getattr(module_dataset, config['near_duplicate_network']['type'])(
        target=True,
        root=config['near_duplicate_network']['args']['data_dir'],
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model) # LOGGING

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    logger.info("device - " + str(device)) # LOGGING

    ############################################################################
    #                                                                          #
    #                           Can be modified (#1)                           #
    #                                                                          #
    # Get nn model features
    with torch.no_grad():
        features, labels = None, []

        logger.info("Start features extracting") # LOGGING

        # Get features and labels
        for i, (data, target) in enumerate(tqdm(image_data_loader, desc="Feature extract by " + config['arch']['type'])):
            data = data.to(device)
            labels += [label.split('.')[0] for label in target]
            output = model(data)

            if features is None:
                features = output
            else:
                features = torch.cat([features, output], dim=0)

    # PCA embedding(to low dimensional)
    n_components_pca = 20
    logger.info("Start dimensionality reduction by pca to " + str(n_components_pca) + "dim") # LOGGING
    embedded_features, explained_variance_ratio = module_arch.DupimageModel.pca_embedding(features, n_components=n_components_pca)

    # computing metrics on test set
    total_metrics = torch.zeros(len(metric_fns))

    # Logging clustering
    logger.info("Start agglomerative clustering") # LOGGING
    log = {
        'pca_dim': n_components_pca,
        'explained_variance_ratio': sum(explained_variance_ratio),
        'result': []
    }

    # Thresholds list
    thresholds = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    #  Agglomerative clustering
    for threshold in tqdm(thresholds, desc="Clustering "):
        # Clustering
        predict_cluster, num_data, _ = module_arch.DupimageModel.agglomerative_clustering(embedded_features, threshold)

        # predict network
        predict_near_duplicate_network = getattr(module_dataset, config['near_duplicate_network']['type'])(
            target=False,
            root=config['near_duplicate_network']['args']['data_dir'],
        )

        # Create predict near-duplicate adjacency matrix
        product_iter = product(range(num_data), range(num_data))
        for i, j in product_iter:
            if predict_cluster[i] == predict_cluster[j]:
                predict_near_duplicate_network.add_edge(labels[i], labels[j])

        # Compute metrics
        total_metrics = torch.zeros(len(metric_fns))
        for i, metric in enumerate(metric_fns):
            total_metrics[i] = metric(predict_near_duplicate_network.get_adj_matrix(), target_near_duplicate_network.get_adj_matrix())

        # Logging
        log['result'].append({
            'threshold': threshold,
            'metrics': {met.__name__: total_metrics[i].item() for i, met in enumerate(metric_fns)}
        })

    # Logging
    logger.info(log) # LOGGING

    # Time logging
    end_time = time()
    logger.info("Experiment time " + str(timedelta(seconds=end_time - start_time)).split(".")[0]) # LOGGING
    
    #                                                                          #
    #                               End of (#1)                                #
    ############################################################################


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='NDIR Research Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)