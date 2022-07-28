import os

# Directory settings
PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATASET_DIR = ''
LOG_DIR = 'logs/'

MODE_PATH = {"train" : '/home/lahj91/SR/SR_training_datasets',
             "test" : '/home/lahj91/SR/SR_testing_datasets'}
TRAINING_DATA_LIST = ["DIV2K", "BSDS200", "General100"]
TEST_DATA_LIST = ["DIV2K", "BSDS100", "Urban100", "Manga109", "Set5", "Set14"]


# Image settings
PIXEL_MAX_VALUE = 255

IMG_NORM_MEAN = [0.406, 0.456, 0.485]  # BGR order
IMG_NORM_STD = [0.225, 0.224, 0.229]  # BGR order

GRAY_COEF = [24.966, 128.553, 65.481]  # BGR order
GRAY_BIAS = 16.