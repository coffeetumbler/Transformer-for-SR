import os

# Directory settings
PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
MATLAB_DATASET_DIR = '/mnt/d/ubuntu/datasets/sr_benchmark/'
LOG_DIR = 'logs/'

MATLAB_MODE_PATH = {"train" : MATLAB_DATASET_DIR + 'Data/train/',
                    "test" : MATLAB_DATASET_DIR + 'Data/valid_ipt/',
                    "evaluation" : MATLAB_DATASET_DIR + 'Data/valid_ipt/',
                    "valid" : MATLAB_DATASET_DIR + 'Data/valid_ipt/'}

MATLAB_DATA_LIST_DIR = MATLAB_DATASET_DIR + 'Data/DataName/'

MATLAB_TRAINING_DATA_LIST = ["DIV2K", "Flickr2K"]
MATLAB_VALID_DATA_LIST = ["Set5", "Set14"]
MATLAB_EVALUATION_DATA_LIST = ["BSD100", "Urban100", "Set5", "Set14"]#manga109 추가 해야 됨

# Image settings
PIXEL_MAX_VALUE = 255

IMG_NORM_MEAN = [0.406, 0.456, 0.485]  # BGR order
IMG_NORM_STD = [0.225, 0.224, 0.229]  # BGR order

GRAY_COEF = [24.966, 128.553, 65.481]  # BGR order
GRAY_BIAS = 16.

# Validation settings
PIXEL_INTERSECTION = 72
IMG_SIZE_UNIT = 24