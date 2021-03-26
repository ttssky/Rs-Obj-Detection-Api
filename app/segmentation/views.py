from app.segmentation import segmentation
from app.segmentation import multiThreadRequest
import os
import sys
import subprocess

# 将segmentation_models_pytorch加入环境变量
# 以便torch.load反序列化时可以加载模型文件
seg_path = os.path.abspath('./app/segmentation')
sys.path.append(seg_path)

import json
import time
import hashlib
import torch
import numpy as np
import rasterio

import geopandas as gpd
from rasterio import Affine
from rasterio.crs import CRS
from pathlib import Path
from glob import glob


from flask import current_app, make_response, request, make_response, \
    jsonify, redirect, current_app, abort
from tqdm import tqdm
from PIL import Image

from config import *
from ..models import create_Idataset_model
from .utils import query_image_meta, Dataset, DataLoader, get_preprocessing
import segmentation_models_pytorch as smp


created_models = {}
@segmentation.route('/segmentation/<category>', methods=['POST'])
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

    #image config
    image_width, image_height, sp_res = query_image_meta(created_models[query_table_name], args['bbox'])
    print("\nres:\n%s" % (sp_res))
    slice_height = args['height'] if args.get('height') is not None else 512
    slice_width = args['width'] if args.get('width') is not None else 512
    slice_overlap = args['slice_overlap'] if args.get('slice_overlap') is not None else 0.3
    sp_slice_height = slice_height * sp_res
    sp_slice_width = slice_width * sp_res

    # 根据请求参数生成uid
    md5 = hashlib.md5()
    md5.update(category.encode('utf-8'))
    for value in args.values():
        md5.update(str(value).encode('utf-8'))
    uid = md5.hexdigest()

    # 文件目录
    base_dir = os.path.join(dl_segmentation_dir, 'buildings')

    file_base_dir = os.path.join(base_dir, 'files', uid)
    if not os.path.exists(file_base_dir):
        os.makedirs(file_base_dir)

    weights_base_dir = os.path.join(base_dir, 'weights', seg_model_ver)
    infer_weights_path = os.path.join(weights_base_dir, seg_name_prefix + \
        '-' + category + '.pth')
    
    infer_images_dir = os.path.join(file_base_dir, 'images')
    if not os.path.exists(infer_images_dir):
        os.makedirs(infer_images_dir)

    # 从heycloud-data-api请求影像瓦片的参数
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

    #并发请求
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

    #-==========================数 据 加 载===========================-

    classes = ['backgroud', category]
    preprocessing_fn = smp.encoders.get_preprocessing_fn(seg_method, 'imagenet')

    test_dataset = Dataset(
        infer_images_dir, 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=classes,
    )
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    #-==========================推 理 过 程============================-

    model = torch.load(infer_weights_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 单张预测二值图tif元信息
    out_meta = {'driver': 'GTiff',
       'dtype': 'uint8',
       'nodata': None,
       'width': slice_width,
       'height': slice_height,
       'count': 1,
       'crs': CRS.from_epsg(3857),
       'transform': None}

    print('[INFO] +=======================推 理 中=========================+\n')

    start = time.time()
    out_dir = os.path.join(file_base_dir, 'tmp_chip_images')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #每个小图片推理，生成二值tif
    for _, batch in enumerate(tqdm(test_loader)):

        name,image  = batch
        x_tensor = image.to(device)
        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_mask = np.argmax(pr_mask, axis=0).astype(np.uint8)

        c = float(name[0].split('__')[-1].split('_')[1])
        f = float(name[0].split('__')[-1].split('_')[0]) + sp_res * slice_height
        transform = Affine(sp_res, 0.0, c, 0.0, -sp_res, f)
        out_meta['transform'] = transform

        out_path = os.path.join(out_dir, name[0] + '.tif')
        out_array = pr_mask.reshape(slice_height, slice_width, -1).transpose(2, 0, 1)
        with rasterio.open(out_path, 'w', **out_meta) as dst:
            dst.write(out_array)
        # Image.fromarray(pr_mask).save(os.path.join(out_dir, name[0] + '.png'))

    print('\n[INFO] total inference time is %ss' % (time.time() - start))

    #-==========================merge为一个大tif并转为polygon============================-

    ims_chip = glob(os.path.join(out_dir, '*.tif'))
    out_merged_path = os.path.join(file_base_dir, 'merge_result.tif')
    exec_cmd = ['gdal_merge.py', '-o ', out_merged_path, '-of GTiff']
    exec_cmd = exec_cmd + ims_chip
    exec_cmd = ' '.join(exec_cmd)

    if os.path.exists(out_merged_path):
        os.remove(out_merged_path)

    if subprocess.call(exec_cmd, shell=True) == 0:
        print('\n[INFO] merge process success\n')
    else:
        print('\n[INFO] merge failed\n')


    #-==========================tif提取polygon========================================-

    out_geojson_path = os.path.join(file_base_dir, 'merge_result.geojson')
    merge_cmd = 'gdal_polygonize.py -f GeoJSON {raster_file} {out_file} \
        label {field_name}'.format(raster_file=out_merged_path, out_file=out_geojson_path, 
        field_name='label')

    if os.path.exists(out_geojson_path):
        os.remove(out_geojson_path)

    if subprocess.call(merge_cmd, shell=True) == 0:
        print('\n[INFO] polygonize process success\n')
    else:
        print('\n[INFO] polygonize failed\n')

    #剔除背景polygon
    df = gpd.read_file(out_geojson_path, driver='GeoJSON')
    df_ = df[df.label == 1]
    df_.to_file(out_geojson_path, driver='GeoJSON')
    
    return df_.to_json()