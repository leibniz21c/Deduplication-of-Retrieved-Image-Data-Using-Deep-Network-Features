import logging
import logging.config
import pandas as pd
from pathlib import Path
from utils import read_json

def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)

def result_logging(logger, result_log, save_dir=None):
    """
    Write expr log and visualize roc curve
    """
    # Create data frame
    df_metrics = pd.DataFrame(result_log)
    df_metrics.to_csv(save_dir / "metric_results.csv")

    # write expr metadata
    logger.info("Test completed...")
    len_hyphen = len(df_metrics.columns)*10 + 5

    # write detail results
    logger.info("-"*len_hyphen)
    logger.info(df_metrics)
    logger.info("-"*len_hyphen)
        
    return df_metrics