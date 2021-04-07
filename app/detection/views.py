from app.detection import detection
from app.detection import multiThreadRequest
import json
import time
import hashlib

from flask import current_app, make_response, request, make_response, \
    jsonify, redirect, current_app, abort

from config import *
from ..models import create_Idataset_model
from .utils import inference, generate_ims_path_txt, \
    query_image_meta, generate_infer_data, generate_net_cfg

created_models = {}
@detection.route('/detection/<category>', methods=['POST'])
def inference_proc(category):
    if request.content_type == 'application/json':
        args = dict(request.get_json(force=True))
    else:
        args = {
            key: value[0] if len(value) == 1 else value
            for key, value in request.form.items()
        }
        args['bbox'] = [float(el) for el in args['bbox'][1:-1].split(',')]

    #idataset model
    query_table_name = 't_%s' % (args['idatasetId'])
    if created_models.get(query_table_name) is None:
        created_models[query_table_name] = create_Idataset_model(query_table_name)

    #get image config
    image_width, image_height, sp_res = query_image_meta(created_models[query_table_name], args['bbox'])
    print("\nres:\n%s" % (sp_res))
    slice_height = args['height'] if args.get('height') is not None else 608
    slice_width = args['width'] if args.get('width') is not None else 608
    slice_overlap = args['slice_overlap'] if args.get('slice_overlap') is not None else 0.3
    sp_slice_height = slice_height * sp_res
    sp_slice_width = slice_width * sp_res

    #netfile config
    batch_size = args['batch_size'] if args.get('batch_size') is not None else 1
    location = args['location'] if args.get('location') is not None else None
    
    if location is not None:
        base_dir = os.path.join(
            dl_detection_dir, 
            category,
            location
        )
       
    else:
        base_dir = os.path.join(
            dl_detection_dir, 
            category
        )

    md5 = hashlib.md5()
    md5.update(category.encode('utf-8'))
    for value in args.values():
        md5.update(str(value).encode('utf-8'))
    uid = md5.hexdigest()

    file_base_dir = os.path.join(base_dir, 'files', uid)
    if not os.path.exists(file_base_dir):
        os.makedirs(file_base_dir)

    weights_base_dir = os.path.join(base_dir, 'weights', det_model_ver)

    infer_images_dir = os.path.join(file_base_dir, 'images')
    if not os.path.exists(infer_images_dir):
        os.makedirs(infer_images_dir)

    infer_txt_path = os.path.join(file_base_dir, 'images_path.txt')
    infer_data_path = os.path.join(file_base_dir, 'inference.data')
    infer_res_dir = os.path.join(file_base_dir, 'results')
    if not os.path.exists(infer_images_dir):
        os.makedirs(infer_images_dir)

    infer_res_txt_dir = os.path.join(infer_res_dir, 'darknet_infer_res')
    if not os.path.exists(infer_res_txt_dir):
        os.makedirs(infer_res_txt_dir)

    infer_classes_names_path = os.path.join(base_dir, category + '.names')
    generate_infer_data(
        infer_data_path,
        infer_txt_path,
        infer_res_dir,
        infer_classes_names_path
    )
    
    if location is not None:
        infer_weights_path = os.path.join(weights_base_dir, det_name_prefix + \
            '-' + category + '-' + location +'.weights')
    else:
        infer_weights_path = os.path.join(weights_base_dir, det_name_prefix + \
            '-' + category +'.weights')

    infer_weights_base_cfg_path = os.path.join(weights_base_dir, 'base.cfg')
    infer_weights_cfg_path = os.path.join(file_base_dir, det_name_prefix + \
        '-' + category + '.cfg')

    generate_net_cfg(
        infer_weights_base_cfg_path,
        infer_weights_cfg_path,
        batch_size,
        slice_height,
        slice_width
    )

    #post process config
    nms_thresh = args['nms_thresh'] if args.get('nms_thresh') is not None else 0.5
    conf_thresh = args['conf_thresh'] if args.get('conf_thresh') is not None else 0.25
   
    #request slice images config
    cfg = current_app.config
    width_min = args['bbox'][0]
    height_min = args['bbox'][1]
    width_max = args['bbox'][2]
    height_max = args['bbox'][3]
    post_params = {
        "bbox": [],
        "bands": [0,1,2],
        "height": slice_height,
        "width": slice_width

    }

    headers = cfg['IMAGE_REQUEST_HEADERS']
    request_url = cfg['IMAGE_REQUEST_HOST'] +  \
                '/heycloud/api/data/idataset' + \
                '/%s' % (args['idatasetId']) + \
                '/extract'


    threads = multiThreadRequest(infer_images_dir,
                        height_min,
                        width_min,
                        height_max,
                        width_max,
                        slice_height,
                        slice_width,
                        sp_slice_height,
                        sp_slice_width,
                        image_height,
                        image_width,
                        slice_overlap,
                        post_params,
                        request_url,
                        headers
                        )
    start = time.time()
   
    for th in threads:
        th.setDaemon(True)
        th.start()
    for t in threads:
        t.join()
    print("\n[INFO] total request time: %ss\n" % round(time.time()-start, 2))


    generate_ims_path_txt(infer_txt_path, infer_images_dir)    

    json = inference(infer_weights_cfg_path, 
              infer_data_path, 
              infer_weights_path, batch_size, 
              infer_txt_path, 
              infer_res_txt_dir,
              conf_thresh, 
              sp_res, 
              args['bbox'], 
              nms_thresh, 
              infer_res_dir)

    return json



