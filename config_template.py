"""
Template of the config.py file to copy in each model directory
Allows us to manage the filepaths to data
"""
class Config:
    ACCELEROMETER_DATA_DIR = '../path/to/accelerometer/'
    UWB_DATA_DIR = '../path/to/uwb/'
    BEHAVIOR_LABELS_DATA_DIR = '../path/to/behavior_labels/'