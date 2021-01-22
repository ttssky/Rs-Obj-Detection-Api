import os

basedir = os.path.abspath(os.path.dirname(__file__))

# image config
SLICE_HEIGHT = 608
SLICE_WIDTH = 608
SLICE_OVERLAP = 0.2
BATCH_SIZE = 1
SP_RES = 0.3435367215073823477
SP_SLICE_HEIGHT = SP_RES * SLICE_HEIGHT
SP_SLICE_WIDTH = SP_RES * SLICE_WIDTH
IMAGE_HEIGHT = 12715
IMAGE_WIDTH = 6825

#path config
IMAGE_PATH_TXT = 'static/data/images_path.txt'
WEIGHTS = 'static/weights/yolov4-sam-mish_best.weights'
NET_CONFIG_FILE = 'static/weights/yolov4_sam_mish_v1.cfg'
INFER_DATA_FILE = 'static/weights/inference.data'
INFER_IMAGE_PATH = 'static/data/images/'
INFER_RES_PATH = 'static/results/'
INFER_RES_TXT = os.path.join(INFER_RES_PATH, 'inference_results.txt')