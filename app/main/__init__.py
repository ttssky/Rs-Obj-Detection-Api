from flask import Blueprint
from .request import multiThreadRequest

main = Blueprint('main', __name__)

from app.main import views, errors
