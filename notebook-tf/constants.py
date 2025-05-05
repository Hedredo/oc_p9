"""Constants for the notebook-torch project
"""
# -*- coding: utf-8 -*-
from pathlib import Path

# FOLDERS VARS
ROOT_FOLDER = Path(__file__).parent.parent.resolve()
IMAGE_FOLDER = ROOT_FOLDER / "data" / "images"
ARTIFACTS_FOLDER = ROOT_FOLDER / "artifacts"
DATASET_PATH = ROOT_FOLDER / "data" / "dataset_cleaned.pkl"

# TRAINING VARS
SEED = 314
SAMPLING = None
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 8
INPUT_RESOLUTION = (224, 224, 3)
