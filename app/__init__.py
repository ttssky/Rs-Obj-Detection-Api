from flask import Flask
from config import config
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

cors = CORS()
db = SQLAlchemy()

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    #TODO其它应用初始化app
    db.init_app(app)
    # ma.init_app(app)
    cors.init_app(app)

    from .detection import detection as detection_blueprint
    app.register_blueprint(detection_blueprint)

    from .segmentation import segmentation as segmentation_blueprint
    app.register_blueprint(segmentation_blueprint)

    #TODO其它蓝图

    return app