from flask import Blueprint
from ..request import multiThreadRequest

detection = Blueprint('detection', __name__)

from app.detection import views, errors
