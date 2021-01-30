from app.main import main
from app.main import multiThreadRequest
import json
import time
# from .utils import *
from flask import current_app, make_response, request, make_response, \
    jsonify, redirect, current_app, abort

from config import *
from ..models import Idataset
from .utils import inference
# @main.before_app_request
# def request_images_from_heycloud():
#     json = jsonify(request.get_json('test'))
#     return None

@main.route('/detection/aircraft', methods=['POST'])
def info():
    if request.content_type == 'application/json':
        args = dict(request.get_json(force=True))
    else:
        args = {
            key: value[0] if len(value) == 1 else value
            for key, value in request.form.items()
        }
        args['bbox'] = [float(el) for el in args['bbox'][1:-1].split(',')]

    SLICE_HEIGHT = args['height'] if args.get('height') is not None else 608
    SLICE_WIDTH = args['width'] if args.get('width') is not None else 608
    SP_SLICE_HEIGHT = SLICE_HEIGHT * SP_RES
    SP_SLICE_WIDTH = SLICE_WIDTH * SP_RES

    cfg = current_app.config
    width_min = args['bbox'][0]
    height_min = args['bbox'][1]
    width_max = args['bbox'][2]
    height_max = args['bbox'][3]
    post_params = {
        "bbox": [],
        "bands": [0,1,2],
        "height": SLICE_HEIGHT,
        "width": SLICE_WIDTH

    }

    headers = cfg['IMAGE_REQUEST_HEADERS']
    request_url = cfg['IMAGE_REQUEST_HOST'] +  \
                '/heycloud/api/data/idataset' + \
                '/37605aef-913e-40b6-8859-822e72a51a19' + \
                '/extract'

    # threads = multiThreadRequest(INFER_IMAGE_PATH,
    #                     height_min,
    #                     width_min,
    #                     height_max,
    #                     width_max,
    #                     SLICE_HEIGHT,
    #                     SLICE_WIDTH,
    #                     SP_SLICE_HEIGHT,
    #                     SP_SLICE_WIDTH,
    #                     IMAGE_HEIGHT,
    #                     IMAGE_WIDTH,
    #                     SLICE_OVERLAP,
    #                     post_params,
    #                     request_url,
    #                     headers
    #                     )
    start = time.time()
   
    # for th in threads:
    #     th.setDaemon(True)
    #     th.start()
    # for t in threads:
    #     t.join()
    print("\n[INFO] total request time: %ss\n" % round(time.time()-start, 2))

    json = inference(NET_CONFIG_FILE, 
              INFER_DATA_FILE, 
              WEIGHTS, BATCH_SIZE, 
              IMAGE_PATH_TXT, 
              INFER_RES_TXT,
              CONF_THRESH, 
              SP_RES, 
              args['bbox'], 
              NMS_THRESH, 
              INFER_RES_PATH)

    return json

@main.route('/test', methods=['POST'])
def test():
    if request.content_type == 'application/json':
        args = dict(request.get_json(force=True))
    else:
        args = {
            key: value[0] if len(value) == 1 else value
            for key, value in request.form.items()
        }
        args['bbox'] = [float(el) for el in args['bbox'][1:-1].split(',')]

    # peter = Idataset.query.filter_by(id='test').first_or_404(description='There is no data with {}'.format('test'))
    # print(Idataset.file)
    abort(404)
    return str(request.accept_mimetypes.accept_json)

