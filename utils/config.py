import os

# Directory settings
PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATASET_DIR = '/home/lahj91/SR/datasets/'
MATLAB_DATASET_DIR = '/data/SR/'
LOG_DIR = 'logs/'

MODE_PATH = {"train" : DATASET_DIR + 'SR_training_datasets/',
             "test" : DATASET_DIR + 'SR_testing_datasets/',
             "valid" : DATASET_DIR + 'SR_testing_datasets/'}
MATLAB_MODE_PATH = {"train" : MATLAB_DATASET_DIR + 'Data/train/',
             "test" : MATLAB_DATASET_DIR + 'Data/test/',
             "valid" : MATLAB_DATASET_DIR + 'Data/test/'}




DATA_LIST_DIR = DATASET_DIR + 'DataName/'
MATLAB_DATA_LIST_DIR = MATLAB_DATASET_DIR + 'Data/DataName/'



TRAINING_DATA_LIST = ["DIV2K", "BSDS200", "General100"]
TEST_DATA_LIST = ["DIV2K", "BSDS100", "Urban100", "Manga109", "Set5", "Set14"]

MATLAB_TRAINING_DATA_LIST = ["DIV2K", "Flickr2K"]
MATLAB_TEST_DATA_LIST = ["DIV2K", "BSDS100", "Urban100", "Manga109", "Set5", "Set14"]##to be updated



# Image settings
PIXEL_MAX_VALUE = 255

IMG_NORM_MEAN = [0.406, 0.456, 0.485]  # BGR order
IMG_NORM_STD = [0.225, 0.224, 0.229]  # BGR order

GRAY_COEF = [24.966, 128.553, 65.481]  # BGR order
GRAY_BIAS = 16.

# Validation settings
PIXEL_INTERSECTION = 17
