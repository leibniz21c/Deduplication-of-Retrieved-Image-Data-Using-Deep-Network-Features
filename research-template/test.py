#
#       Only binary classification, with thresholding cosine distance
#       to preprocessed dataset
#
import argparse
from time import time
from datetime import timedelta
from tqdm import tqdm
import numpy as np
import torch
import data_loaders.data_loaders as module_data_loader
import logger.visualization as module_visualization
import model.metric as module_metric
from parse_config import ConfigParser
from utils import prepare_device
from logger import result_logging


def main(config):
    # Get logger
    logger = config.get_logger(config["name"])

    # Time logging
    start_time = time()

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    logger.info("Device list : " + str(device_ids))  # LOGGING

    # Get features dataloader
    pairs_data_loader = getattr(
        module_data_loader, config["pairs_data_loader"]["type"]
    )(
        **config["pairs_data_loader"]["args"],
    )

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    # cosine distance based binary classification
    cosim = torch.nn.CosineSimilarity(dim=1).to(device)
    codist = lambda u, v: 1 - cosim(u, v)

    # Thresholds list
    if config["distance_thresholds"]["logscale"]:
        thresholds = np.logspace(
            config["distance_thresholds"]["start"],
            config["distance_thresholds"]["end"],
            num=config["distance_thresholds"]["num_threshold"],
            endpoint=True,
        )
    else:
        thresholds = np.linspace(
            config["distance_thresholds"]["start"],
            config["distance_thresholds"]["end"],
            num=config["distance_thresholds"]["num_threshold"],
            endpoint=True,
        )

    # Test phase
    result_log = []

    logger.info(
        "Start near duplicate classification for each threshold ("
        + str(len(thresholds))
        + " times)"
    )

    # Metrics buffer
    total_metrics = torch.zeros(len(metric_fns))
    for threshold in tqdm(thresholds):
        predicts = None
        labels = None
        for (data1, data2), label in pairs_data_loader:
            predict = (codist(data1[0].to(device), data2[0].to(device)) <= threshold).int()
            label = label.to(device).int()

            # binary classification respect to data pairs
            if predicts == None:
                predicts = predict
                labels = label
            else:
                predicts = torch.cat([predicts, predict], dim=0)
                labels = torch.cat([labels, label], dim=0)
        
        # Compute metrics
        for i, metric in enumerate(metric_fns):
            total_metrics[i] = metric(predicts, labels)

        # Logging
        result = {
            met.__name__: total_metrics[i].item() for i, met in enumerate(metric_fns)
        }
        result.update({"threshold": threshold})
        result_log.append(result)

    # With no result
    if len(result_log) == 0:
        end_time = time()
        logger.info(
            "Experiment time "
            + str(timedelta(seconds=end_time - start_time)).split(".")[0]
        )  # LOGGING

    # Logging
    df_result = result_logging(logger, result_log, save_dir=config.save_dir)

    # Visualizatrion
    [
        getattr(
            module_visualization,
            fn,
        )(df_result, save_dir=config.save_dir / (fn + ".png"), config=config)
        for fn in config["visualization"]
    ]

    # Time logging
    end_time = time()
    logger.info(
        "Experiment time " + str(timedelta(seconds=end_time - start_time)).split(".")[0]
    )  # LOGGING

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="NDIR Research Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)
    main(config)
