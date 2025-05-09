"""Constants for the api"""
# -*- coding: utf-8 -*-
from pathlib import Path

# FOLDERS VARS
ROOT_FOLDER = Path(__file__).parent.parent.resolve()
ARTIFACTS_FOLDER = ROOT_FOLDER / "artifacts"

# VARS
MODEL_PATH = ARTIFACTS_FOLDER / "20250506-162559_MambaVision-T-1K_unfreezed.pth"
INPUT_RESOLUTION = (3, 224, 224)
MODEL_CARD = "nvidia/MambaVision-T-1K"
LABELS = [
    "Baby Care",
    "Beauty and Personal Care",
    "Computers",
    "Home Decor & Festive Needs",
    "Home Furnishing",
    "Kitchen & Dining",
    "Watches",
]
N_CLASSES = len(LABELS)
LABEL2IDX = {label: i for i, label in enumerate(LABELS)}

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
