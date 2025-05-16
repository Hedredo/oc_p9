"""Constants for the frontend
"""
# -*- coding: utf-8 -*-
from pathlib import Path

# FOLDERS VARS
ROOT_FOLDER = Path(__file__).parent.parent.resolve()
CLOUD_FOLDER = Path(__file__).parent.resolve()
IMAGE_FOLDER = ROOT_FOLDER / "data" / "images"
DATASET_PATH = ROOT_FOLDER / "data" / "dataset_cleaned.pkl"
API_URL = "http://valid-flowing-mantis.ngrok-free.app"

# VARS
INPUT_RESOLUTION = (3, 224, 224)
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
