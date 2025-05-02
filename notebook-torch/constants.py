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
SAMPLING = 32
TEST_SIZE = 0.15
VAL_SIZE = 0.15
BATCH_SIZE = 8
INPUT_RESOLUTION = (3, 224, 224)

# MAMBAVISION DICT {"HUGGINFACE Model Card": feature_extractor_size}
MAMBA_HIDDEN_SIZES = {
    "nvidia/MambaVision-T-1K": 640,
    "nvidia/MambaVision-T2-1K": 640,
    "nvidia/MambaVision-S-1K": 768,
    "nvidia/MambaVision-B-1K": 1024,
    "nvidia/MambaVision-L-1K": 1568,
    "nvidia/MambaVision-B-21K": 1024,
    "nvidia/MambaVision-L-21K": 1568,
}