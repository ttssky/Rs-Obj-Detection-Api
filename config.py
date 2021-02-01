import os

basedir = os.path.abspath(os.path.dirname(__file__))

request_image_host = 'http://192.168.31.17'
port = 9000

method = 'yolov4'
model_ver = 'v0.0.1'
name_prefix = method + '-' + model_ver
dl_detection_dir = '/home/geohey/volumes/dl-detection/'
dl_segmentation_dir = '/home/geohey/volumes/dl-segmentation/'

uid = 'test'
# image config


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    IMAGE_REQUEST_HOST = "http://192.168.31.17:9000"
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