import argparse
import logging
from datetime import datetime
from glob import glob
from os.path import join, dirname, abspath

import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


__dirname__ = dirname(__file__)

import pandas as pd


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Copy evaluation data")
    parser.add_argument('--output_path', type=str, default='human_20250501_131304.npz', help='Output file name')
    args = parser.parse_args()

    src = abspath(args.output_path)
    
    logger.info(f"Loading dataset from {src}")

    # load npz file
    dataset = np.load(src, allow_pickle=True)

    logger.info(f"Loaded dataset with keys: {dataset} ({type(dataset)})")

    for key, val in dataset.items():
        logger.info(f"Key: {key}, Val: {val.shape}")
