from flask import Blueprint
from ..request import multiThreadRequest

segmentation = Blueprint('segmentation', __name__)

from app.segmentation import views, errors
