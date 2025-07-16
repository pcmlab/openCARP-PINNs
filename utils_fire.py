import yaml
import logging
import os
from typing import Generator, Any
import torch


def get_config(config_path: str) -> dict:
    """Load config file from give path

    Parameters
    ----------
    config_path : str
        Path to config file

    Returns
    -------
    dict
        dict-like config.
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_logger() -> logging.Logger:
    """Create and setup logger. Always in DEBUG mode.

    Returns
    -------
    logging.Logger
        Logger object.
    """

    #  setup logger
    logging.getLogger('cmai').handlers.clear()
    logger = logging.getLogger('cmai')
    logger_handler = logging.StreamHandler()
    logger_handler.setLevel(logging.DEBUG)
    logger_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s: %(message)s'))
    logger.addHandler(logger_handler)
    logger.setLevel(logging.DEBUG)

    return logger


def make_result_dir(prefix: str = '') -> str:
    """Make directory to save all results.

    Parameters
    ----------
    prefix : str
        Directory prefix part before number part
    n_folds : int
        Number of fords for cross-validation training.
        Every fold has own result directory.

    Returns
    -------
    str
        Path to result dir
    """

    res_dir_root = os.path.join('./results', prefix)
    os.makedirs(res_dir_root, exist_ok=True)
    return res_dir_root


