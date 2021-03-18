from app.main import main
from flask import request, jsonify, Response
import json

@main.app_errorhandler(404)
def infernal_server_error(error):
    print(error)
    return 404