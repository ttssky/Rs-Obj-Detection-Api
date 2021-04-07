from app.detection import detection
from flask import request, jsonify, Response
import json

@detection.app_errorhandler(404)
def infernal_server_error(error):
    print(error)
    return 404