import os

basedir = os.path.abspath(os.path.dirname(__file__))

# image config
# SLICE_HEIGHT = 608
# SLICE_WIDTH = 608
SLICE_OVERLAP = 0.2
BATCH_SIZE = 1
SP_RES = 0.3435367215073823477
CONF_THRESH = 0.25
NMS_THRESH = 0.5
# SP_SLICE_HEIGHT = SP_RES * SLICE_HEIGHT
# SP_SLICE_WIDTH = SP_RES * SLICE_WIDTH
IMAGE_HEIGHT = 12715
IMAGE_WIDTH = 6825

#path config
IMAGE_PATH_TXT = '/data/htwy_proj/aircraft-object-detection-api/htwy-obj-detection-api/static/data/images_path.txt'
WEIGHTS = '/data/htwy_proj/aircraft-object-detection-api/htwy-obj-detection-api/static/weights/yolov4-sam-mish_best.weights'
NET_CONFIG_FILE = '/data/htwy_proj/aircraft-object-detection-api/htwy-obj-detection-api/static/weights/yolov4_sam_mish_v1.cfg'
INFER_DATA_FILE = '/data/htwy_proj/aircraft-object-detection-api/htwy-obj-detection-api/static/weights/inference.data'
INFER_IMAGE_PATH = '/data/htwy_proj/aircraft-object-detection-api/htwy-obj-detection-api/static/data/images/'
INFER_RES_PATH = '/data/htwy_proj/aircraft-object-detection-api/htwy-obj-detection-api/static/results/'
INFER_RES_TXT = os.path.join(INFER_RES_PATH, 'inference_results')

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    IMAGE_REQUEST_HOST = "http://192.168.31.17:9001"
    IMAGE_REQUEST_HEADERS = {
        "x-heycloud-admin-session": "tFn8aWIgWHnYTCO/NR/r2OK4wef96gtC",
        "Content-Type":"application/json"
    }

    @staticmethod
    def init_app(app):
        pass


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'postgresql+psycopg2://postgres:postgres@192.168.31.17:15432/heycloud'


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or \
        'sqlite://'


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'data.sqlite')


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,

    'default': DevelopmentConfig
}