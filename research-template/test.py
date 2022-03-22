import argparse
import os
from time import time
from datetime import timedelta
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import PCA
import torch
import data_loaders.data_loaders as module_data_loader
import logger.visualization as module_visualization
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import prepare_device
from logger import result_logging


def main(config):
    # Get logger
    logger = config.get_logger(config['name'])

    # Time logging
    start_time = time()

     # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # Check preprocessed checkpoint features mode
    if not config["pairs_data_loader"]["args"]["preprocessed_path"]:
        # build model architecture
        model = config.init_obj('feature_arch', module_arch)
        logger.info(model) # LOGGING
        model.eval()
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        else:
            model = model.to(device)
        logger.info("Device list : " + str(device_ids)) # LOGGING

        # Get ready to PCA transformation 
        images_data_loader = getattr(
            module_data_loader, 
            config['pca_transform_data_loader']['type'])(
                batch_size=config['pca_transform_data_loader']['args']['batch_size'],
                data_dir=config['pca_transform_data_loader']['args']['data_dir'],
                num_workers=config['pca_transform_data_loader']['args']['num_workers'],
                preprocessed_path=config["pairs_data_loader"]["args"]["preprocessed_path"])
        logger.info("Get ready to PCA transformation") # LOGGING
        with torch.no_grad():
            features = None
            # Get features and labels
            for i, (data, _) in enumerate(tqdm(images_data_loader, desc="Feature extract by " + config['feature_arch']['type'])):
                data = data.to(device)
                output = model(data)

                if features is None:
                    features = output
                else:
                    features = torch.cat([features, output], dim=0)

            features = features.cpu().numpy()
            features = PCA(n_components=config["pca_transform_data_loader"]["dim"]).fit_transform(features)
        
        if not os.path.isdir(config['pca_transform_data_loader']['args']['data_dir'] + "/preprocessed"):
            os.mkdir(config['pca_transform_data_loader']['args']['data_dir'] + "/preprocessed")

        np.save(config['pca_transform_data_loader']['args']['data_dir'] + "preprocessed/" + config['pca_transform_data_loader']['args']['data_dir'].split('/')[-1] + config['name'] + "-features.npy", features)
        preprocessed_path = config['pca_transform_data_loader']['args']['data_dir'] + "preprocessed/" + config['pca_transform_data_loader']['args']['data_dir'].split('/')[-1] + config['name'] + "-features.npy"

        del features, images_data_loader
        torch.cuda.empty_cache()
    else:
        preprocessed_path = "/datasets/california-nd/preprocessed/dupimage-california-nd-gt0.5-features.npy"

    # setup image data_loader instances
    logger.info("Preprocessed data path : " + preprocessed_path)
    pairs_data_loader = getattr(
        module_data_loader, 
        config['pairs_data_loader']['type'])(
            batch_size=config['pairs_data_loader']['args']['batch_size'],
            data_dir=config['pairs_data_loader']['args']['data_dir'],
            corr_threshold=config['pairs_data_loader']['args']['corr_threshold'],
            num_workers=config['pairs_data_loader']['args']['num_workers'],
            preprocessed_path=preprocessed_path,
            nnd_approx_equal=config['pairs_data_loader']['args']['nnd_approx_equal'])
            
    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # cosine distance based binary classification
    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    codist = lambda u, v: 1 - cosim(u, v)

    # Thresholds list
    if config["distance_thresholds"]["logscale"]:
        thresholds = np.logspace(
            config["distance_thresholds"]["start"],
            config["distance_thresholds"]["end"],
            num=config["distance_thresholds"]["num_threshold"],
            endpoint=True
        )
    else:
        thresholds = np.linspace(
            config["distance_thresholds"]["start"], 
            config["distance_thresholds"]["end"], 
            num=config["distance_thresholds"]["num_threshold"],
            endpoint=True
        )

    # Test phase
    result_log = []

    logger.info("Start near duplicate classification for each threshold (" + str(len(thresholds)) + " times)")
    for threshold in tqdm(thresholds):
        predicts = None
        labels = None
        total_metrics = torch.zeros(len(metric_fns))
        with torch.no_grad():
            for i, ((data1, data2), label) in enumerate(pairs_data_loader):
                data1, data2, label = data1.to(device), data2.to(device), label.to(device)

                # binary classification respect to data pairs
                if predicts == None:
                    predicts = codist(data1, data2) <= threshold
                    labels = label
                else:
                    predicts = torch.cat([predicts, (codist(data1, data2) <= threshold)], dim=0)
                    labels = torch.cat([labels, label], dim=0)

            # Compute metrics
            for i, metric in enumerate(metric_fns):
                total_metrics[i] = metric(predicts, labels)

        # Logging
        result = {met.__name__: total_metrics[i].item() for i, met in enumerate(metric_fns)}
        result.update({'threshold': threshold})
        result_log.append(result)
    
    
    # With no result
    if len(result_log) == 0:
        end_time = time()
        logger.info("Experiment time " + str(timedelta(seconds=end_time - start_time)).split(".")[0]) # LOGGING

    # Logging
    df_result = result_logging(logger, result_log, save_dir=config.save_dir)

    # Visualizatrion
    [getattr(module_visualization, fn,)(df_result, save_dir=config.save_dir / (fn + ".png"), config=config) for fn in config['visualization']]

    # Time logging
    end_time = time()
    logger.info("Experiment time " + str(timedelta(seconds=end_time - start_time)).split(".")[0]) # LOGGING


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