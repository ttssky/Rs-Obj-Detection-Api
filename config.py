import os

basedir = os.path.abspath(os.path.dirname(__file__))

image_request_heycloud_host = 'http://192.168.31.17' #获取预测影像的heycloud-data-api地址
port = 9000 #获取预测影像的heycloud-data-api的端口号

#存储影像信息的sqlalchemy格式pg地址
db_sqlalchemy_addr = 'postgresql+psycopg2://postgres:postgres@192.168.31.17:15432/heycloud'

#目标检测预测模型信息
det_method = 'yolov4' #目标检测方法
det_model_ver = 'v0.0.1'
det_name_prefix = det_method + '-' + det_model_ver
dl_detection_dir = '/home/geohey/volumes/dl-detection/'

#语义分割模型信息
seg_method = 'efficientnet-b4' #语义分割方法
seg_model_ver = 'v0.0.1'
seg_name_prefix = seg_method + '-' + seg_model_ver
dl_segmentation_dir = '/home/geohey/volumes/dl-segmentation/'

#Flask app配置

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    IMAGE_REQUEST_HOST = image_request_heycloud_host + ':' + str(port)
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