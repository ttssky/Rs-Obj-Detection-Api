import os
from glob import glob
from . import darknet
import time
import cv2
import numpy as np
import random
import time

from .core import add_geo_coords_to_df
from .core import post_process_create_df
from .core import refine_df

def load_images(images_path):

    if images_path.split('.')[-1] == 'txt':
        with open(images_path, 'r') as f:
            return f.read().splitlines()

    else:
        raise ValueError

def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    print("img path is {}".format(image_path))
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)
    
    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    
    detections, nbox = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections, nbox, darknet_image.w, darknet_image.h

def generate_infer_res_txt(txt_path, image_path, dets, nbox, class_names, image_w, image_h):

    for i in range(nbox):
        #det[('3', '98.21', (8.879005432128906, 54.7884407043457, 18.473737716674805, 107.47196960449219))]
        #det 3表示所属类别，98.21为置信度 最后为x y w h
        xmin = dets[i][2][0] - dets[i][2][2] / 2. + 1
        xmax = dets[i][2][0] + dets[i][2][2] / 2. + 1
        ymin = dets[i][2][1] - dets[i][2][3] / 2. + 1
        ymax = dets[i][2][1] + dets[i][2][3] / 2. + 1

        if (xmin < 1): xmin = 1
        if (ymin < 1): ymin = 1
        if (xmax > image_w): xmax = image_w
        if (ymax > image_h): ymax = image_h


        for idx, name in enumerate(class_names):
            if (dets[i][0] == name):
                class_txt = os.path.join(txt_path, name + '.txt') 
                
                with open(class_txt, 'a') as f:
                    bbox_data = image_path + ' ' + str(dets[i][1]) \
                        + ' ' + str(xmin) + ' '+ str(ymin) + ' ' + str(xmax) + ' ' + str(ymax)
                    f.write(bbox_data + '\n')

def post_process(
             slice_sizes=[608],
            #  testims_dir_tot='', #测试图像的路径
             infer_classes_files_dir='', #darknet输出的txt,每个类别对应一个txt
             test_slice_sep='__',
             edge_buffer_test=3, #距离边界的阈值
             max_edge_aspect_ratio=3,
             test_box_rescale_frac=1.0,
             sp_res=0,
             bbox=[],
             rotate_boxes=False,
             test_add_geo_coords=True,
             verbose=False
             ):
    

    # post-process
    # df_tot = post_process_yolt_test_create_df(args)
    infer_classes_files = glob(os.path.join(infer_classes_files_dir, '*.txt'))
    df_tot = post_process_create_df(
        infer_classes_files,
        # testims_dir_tot=testims_dir_tot,
        slice_sizes=slice_sizes,
        slice_sep=test_slice_sep,
        edge_buffer_test=edge_buffer_test,
        max_edge_aspect_ratio=max_edge_aspect_ratio,
        test_box_rescale_frac=test_box_rescale_frac,
        sp_res=sp_res,
        bbox=bbox,
        rotate_boxes=rotate_boxes)


    df_tot.to_csv('test.csv', index=False)
    print(type(df_tot))
    return df_tot

def get_nms_add_geos_geojson(df,
                            save_dir,
                            nms_thresh, 
                            conf_thresh=0.8, 
                            create_geojson=True, 
                            Proj_str="epsg:3857", 
                            verbose=False):

    df_res = refine_df(
        df=df, 
        nms_overlap_thresh=nms_thresh, 
        retain_thresh=conf_thresh
    )

    df_, json = add_geo_coords_to_df(
        df, 
        create_geojson=create_geojson, 
        Proj_str=Proj_str
    )
    output_csv = os.path.join(save_dir, 'results.csv')
    output_geojson = os.path.join(save_dir, 'results.geojson')

    df_.to_csv(output_csv, index=False)
    json.to_file(output_geojson, driver="GeoJSON")
    return json.to_json()

def inference(net_config_file,
              data_file,
              weights,
              batch_size,
              images_txt_path,
              infer_res_txt_dir,
              conf_thresh,
              sp_res,
              bbox,
              nms_thresh,
              results_dir
              ):

    random.seed(2222)
    network, class_names, class_colors = darknet.load_network(
        net_config_file,
        data_file,
        weights,
        batch_size=batch_size
    )
    images_path = load_images(images_txt_path)
    for name in class_names:
        txt = os.path.join(infer_res_txt_dir, str(name) + '.txt')
        if os.path.exists(txt): os.remove(txt)
    # print(images_path)
    index = 0
    while True:
    
        prev_time = time.time()
        if index >= len(images_path):
            break
        image_name = images_path[index]
        try:
            images, detections, nbox, w, h = image_detection(
                image_name, network, class_names, class_colors, conf_thresh
            )
        except:
            index += 1
            continue
        print(detections, nbox)

        generate_infer_res_txt(infer_res_txt_dir, image_name, 
                        detections, nbox, class_names, w, h)
        darknet.print_detections(detections)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        index += 1
 
    df_post_process = post_process(
                                            infer_classes_files_dir=infer_res_txt_dir,
                                            sp_res=sp_res,
                                            bbox=bbox,
                                            edge_buffer_test=3
                                            )
    json = get_nms_add_geos_geojson(df=df_post_process, 
    save_dir=results_dir, 
    nms_thresh=nms_thresh,
    conf_thresh=conf_thresh)

    return json
# if __name__ == "__main__":
#     inference(devconfig)  