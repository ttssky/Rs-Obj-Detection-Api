import os
import time

import gdal
import geopandas as gpd
import shapely
import shapely.geometry
import pyproj
import affine as af
import pandas as pd
import numpy as np
import cv2
import rasterio as rio

from .non_max_suppression import non_max_suppression


__all__ = ["post_process_create_df", "refine_df"]

def post_process_create_df(infer_classes_files,
                                     slice_sizes=[608],
                                     slice_sep='__',
                                     edge_buffer_test=0,
                                     max_edge_aspect_ratio=4,
                                     sp_res=0,
                                     bbox=[],
                                     test_box_rescale_frac=1.0,
                                     rotate_boxes=False,
                                     verbose=False):
    """
    将darknet预测后的类别目标框坐标txt转化为dataframe.

    Arguments
    ---------
    infer_classes_files : list
        预测后的类别文件列表 (e.g. [class1.txt, class2.txt]).
    slice_sizes : list
        原始影像裁剪窗口大小，默认608
    slice_sep : str
        提取目标框所属裁剪图片的分割符号，默认“__”
    edge_buffer_test : int
        如果某个目标框离图片边缘的距离小于这个值，将会被忽略，默认0
    max_edge_aspect_ratio : int
        目标框长宽的最大比例，大于这个比例，将会忽略
    sp_res : float
        影像的空间分辨率
    bbox : list
        整个影像的范围
    test_box_rescale_frac : float
        目标框坐标的范围
    rotate_boxes : boolean
        是否旋转
    verbose : boolean
        是否print
    Returns
    -------
    df : pandas dataframe
        返回一个dataframe，记录了每个目标框的全局坐标，但是没有经过nms
    """

    # parse out files, create df
    df_tot = []

    for i, vfile in enumerate(infer_classes_files):

        test_base_string = '"test_file: ' + str(vfile) + '\n"'
        print(test_base_string[1:-2])
    
        cat = vfile.split('/')[-1].split('.')[0]
        # 将txt转化为dataframe
        print(vfile)
    
        df = pd.read_csv(vfile, sep=' ', names=['Loc_Tmp', 'Prob',
                                                'Xmin', 'Ymin', 'Xmax',
                                                'Ymax'])
        #保存类别信息，并根据空间分辨率将预测的像素坐标转化为空间坐标
        df['Category'] = len(df) * [cat]
        df_min = pd.DataFrame(df['Ymin'])
        df_max = pd.DataFrame(df['Ymax'])
        df['Xmin'] = df['Xmin'] * sp_res
        df['Ymin'] = (slice_sizes[0] - df_max) * sp_res
        df['Xmax'] = df['Xmax'] * sp_res
        df['Ymax'] = (slice_sizes[0] - df_min) * sp_res

        # augment
        df = augment_df(df,
                        slice_sizes=slice_sizes,
                        slice_sep=slice_sep,
                        edge_buffer_test=edge_buffer_test,
                        max_edge_aspect_ratio=max_edge_aspect_ratio,
                        sp_res=sp_res,
                        bbox=bbox,
                        test_box_rescale_frac=test_box_rescale_frac,
                        rotate_boxes=rotate_boxes)

        # append to total df
        if i == 0:
            df_tot = df
        else:
            df_tot = df_tot.append(df, ignore_index=True)

    return df_tot


def augment_df(df,
            slice_sizes=[608],
            slice_sep='__',
            edge_buffer_test=0,
            max_edge_aspect_ratio=4,
            test_box_rescale_frac=1.0,
            sp_res=0.34,
            bbox=[],
            rotate_boxes=False,
            verbose=False):

    """
    为dataframe添加目标框的全局坐标.

    Arguments
    ---------
    df : pandas dataframe
        每个目标框图片内坐标的dataframe
    slice_sizes : list
        裁剪窗口大小
    slice_sep : str
        小图片名称与记录小图片全局位置的分隔符号
    edge_buffer_test : int
    
    max_edge_aspect_ratio : int

    test_box_rescale_frac : float

    bbox : list
    rotate_boxes : boolean
    verbose : boolean

    Returns
    -------
    df : pandas dataframe
        添加全局坐标
    """

    extension_list = ['.png', '.tif', '.TIF', '.TIFF', '.tiff', '.JPG',
                      '.jpg', '.JPEG', '.jpeg']
    t0 = time.time()
    print("Augmenting dataframe of initial length:", len(df), "...")

    im_roots, im_locs = [], []
    # for j, f in enumerate(df['Image_Root_Plus_XY'].values):
    for j, loc_tmp in enumerate(df['Loc_Tmp'].values):

        if (j % 10000) == 0:
            print(j)

        f = loc_tmp.split('/')[-1]
        ext = f.split('.')[-1]
        # 获取这一影像的名称或标识ID
        if slice_sizes[0] > 0:
            im_root_tmp = f.split(slice_sep)[0]
            xy_tmp = f.split(slice_sep)[-1]

        im_locs.append(xy_tmp)

        if '.' not in im_root_tmp:
            im_roots.append(im_root_tmp + '.' + ext)
        else:
            im_roots.append(im_root_tmp)

    if verbose:
        print("loc_tmp[:3]", df['Loc_Tmp'].values[:3])
        print("im_roots[:3]", im_roots[:3])
        print("im_locs[:3]", im_locs[:3])

    df['Image_ID'] = im_roots
    df['Slice_XY'] = im_locs
    # 获取小图片对应坐标
    df['Upper'] = [float(sl.split('_')[0]) for sl in df['Slice_XY'].values]
    df['Left'] = [float(sl.split('_')[1]) for sl in df['Slice_XY'].values]
    df['Height'] = [float(sl.split('_')[2]) for sl in df['Slice_XY'].values]
    df['Width'] = [float(sl.split('_')[3]) for sl in df['Slice_XY'].values]
    df['Pad'] = [float(sl.split('_')[4].split('.')[0])
                 for sl in df['Slice_XY'].values]
    df['Im_Width'] = [float(sl.split('_')[5].split('.')[0])
                      for sl in df['Slice_XY'].values]
    df['Im_Height'] = [float(sl.split('_')[6].split('.')[0])
                       for sl in df['Slice_XY'].values]

    if verbose:
        print("  Add in global location of each row")
    # 根据小图片坐标和目标框局部坐标获取全局坐标
    if slice_sizes[0] > 0:
        x0l, x1l, y0l, y1l = [], [], [], []
        bad_idxs = []
        for index, row in df.iterrows():
            bounds, coords = get_global_coords(
                row,
                edge_buffer_test=edge_buffer_test,
                max_edge_aspect_ratio=max_edge_aspect_ratio,
                test_box_rescale_frac=test_box_rescale_frac,
                sp_res=sp_res,
                bbox=bbox,
                rotate_boxes=rotate_boxes)
            if len(bounds) == 0 and len(coords) == 0:
                bad_idxs.append(index)
                [xmin, xmax, ymin, ymax] = 0, 0, 0, 0
            else:
                [xmin, xmax, ymin, ymax] = bounds
            x0l.append(xmin)
            x1l.append(xmax)
            y0l.append(ymin)
            y1l.append(ymax)
        df['Xmin_Glob'] = x0l
        df['Xmax_Glob'] = x1l
        df['Ymin_Glob'] = y0l
        df['Ymax_Glob'] = y1l

    else:
        df['Xmin_Glob'] = df['Xmin'].values
        df['Xmax_Glob'] = df['Xmax'].values
        df['Ymin_Glob'] = df['Ymin'].values
        df['Ymax_Glob'] = df['Ymax'].values
        bad_idxs = []

    # 移除有问题的目标框
    if len(bad_idxs) > 0:
        print("removing bad idxs near junctions:", bad_idxs)
        df = df.drop(df.index[bad_idxs])

    print("Time to augment dataframe of length:", len(df), "=",
          time.time() - t0, "seconds")
    return df

def get_global_coords(row,
                      edge_buffer_test=0,
                      max_edge_aspect_ratio=4,
                      test_box_rescale_frac=1.0,
                      sp_res=0.34,
                      bbox=[],
                      rotate_boxes=False):
    """
    获取每个目标框的全局坐标.

    Arguments
    ---------
    row : pandas dataframe row
        具有小图片坐标，目标框局部坐标的dataframe row
    edge_buffer_test : int
    max_edge_aspect_ratio : int
    test_box_rescale_frac : float
    bbox : list
    rotate_boxes : boolean

    Returns
    -------
    bounds, coords : tuple
        bounds = [xmin, xmax, ymin, ymax] 目标框范围
        coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]] 目标框四角坐标
    """

    xmin0, xmax0 = row['Xmin'], row['Xmax']
    ymin0, ymax0 = row['Ymin'], row['Ymax']
    upper, left = row['Upper'], row['Left']
    sliceHeight, sliceWidth = row['Height'], row['Width']
    vis_w, vis_h = row['Im_Width'], row['Im_Height']
    pad = row['Pad']

    if edge_buffer_test > 0:
        # 如果目标框在edge_buffer_test之内，则返回空
        if ((float(xmin0) < edge_buffer_test) or
            (float(xmax0) > (sliceWidth * sp_res - edge_buffer_test)) or
            (float(ymin0) < edge_buffer_test) or
                (float(ymax0) > (sliceHeight * sp_res - edge_buffer_test))):
            return [], []
        # 如果buffer之内，而且长宽比大于阈值，返回空
        elif ((float(xmin0) < edge_buffer_test) or
                (float(xmax0) > (sliceWidth * sp_res - edge_buffer_test)) or
                (float(ymin0) < edge_buffer_test) or
                (float(ymax0) > (sliceHeight * sp_res - edge_buffer_test))):
            # 计算长宽比
            dx = xmax0 - xmin0
            dy = ymax0 - ymin0
            if (1.*dx/dy > max_edge_aspect_ratio) \
                    or (1.*dy/dx > max_edge_aspect_ratio):
                return [], []
    dx = xmax0 - xmin0
    dy = ymax0 - ymin0
    if (1.*dx/dy > max_edge_aspect_ratio) \
            or (1.*dy/dx > max_edge_aspect_ratio):
            return [], []

    # 计算目标框全局坐标
    print('bbox:\n',bbox)
    xmin = max(bbox[0], float(float(xmin0)+left - pad))
    xmax = min(bbox[2], float(float(xmax0)+left - pad))
    ymin = max(bbox[1], float(float(ymin0)+upper - pad))
    ymax = min(bbox[3], float(float(ymax0)+upper - pad))

    # 如果需要rescale目标框size
    if test_box_rescale_frac != 1.0:
        dl = test_box_rescale_frac
        xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
        dx = dl*(xmax - xmin) / 2
        dy = dl*(ymax - ymin) / 2
        x0 = xmid - dx
        x1 = xmid + dx
        y0 = ymid - dy
        y1 = ymid + dy
        xmin, xmax, ymin, ymax = x0, x1, y0, y1

    # 如果需要，旋转目标框
    if rotate_boxes:
        vis = cv2.imread(row['Image_Path'], 1)  # color
        gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
        canny_edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        coords = _rotate_box(xmin, xmax, ymin, ymax, canny_edges)

    # 计算bounds和coords
    bounds = [xmin, xmax, ymin, ymax]
    coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    # 确保没有负值
    if np.min(bounds) < 0:
        print("part of bounds < 0:", bounds)
        print(" row:", row)
        return
    if (xmax > bbox[2]) or (ymax > bbox[3]):
        print("part of bounds > image size:", bounds)
        print(" row:", row)
        return

    return bounds, coords


def refine_df(df, groupby='Image_ID',
              cats_to_ignore=[],
              use_weighted_nms=True,
              nms_overlap_thresh=0.5,  retain_thresh=0.25,
              verbose=True):
    """
    移除低于置信度的目标框，并且根据阈值进行nms.

    Arguments
    ---------
    df : pandas dataframe
        augment后的dataframe
    groupby : str
    groupby_cat : str
    cats_to_ignore : list
        忽略的类别
    use_weighted_nms : boolean
        使用weighted_nms，默认true
    nms_overlap_thresh : float
        nms阈值，默认0.5
    retain_thresh : float
        置信度阈值
    verbose : boolean
        是否打印
    Returns
    -------
    df_tot : pandas dataframe
        经过置信度筛选和nms后的目标框
    """

    print("Running refine_df()...")
    t0 = time.time()

    # group by imageid or title
    group = df.groupby(groupby)
    count = 0
    print_iter = 1
    df_idxs_tot = []
    for i, g in enumerate(group):

        img_loc_string = g[0]
        data_all_classes = g[1]

        if (i % print_iter) == 0 and verbose:
            print(i+1, "/", len(group), "Processing image:", img_loc_string)
            print("  num boxes:", len(data_all_classes))

        data = data_all_classes.copy()
        # 筛选需要忽略的类别
        if len(cats_to_ignore) > 0:
            data = data[~data['Category'].isin(cats_to_ignore)]
        df_idxs = data.index.values
        scores = data['Prob'].values



        xmins = data['Xmin_Glob'].values
        ymins = data['Ymin_Glob'].values
        xmaxs = data['Xmax_Glob'].values
        ymaxs = data['Ymax_Glob'].values

        # 忽略低于置信度阈值的目标框
        high_prob_idxs = np.where(scores >= retain_thresh)
        scores = scores[high_prob_idxs]
        xmins = xmins[high_prob_idxs]
        xmaxs = xmaxs[high_prob_idxs]
        ymins = ymins[high_prob_idxs]
        ymaxs = ymaxs[high_prob_idxs]
        df_idxs = df_idxs[high_prob_idxs]

        boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)

        if verbose:
            print("len boxes:", len(boxes))

            ###########
            # 执行NMS
        if nms_overlap_thresh > 0:
            # Try nms with pyimagesearch algorighthm
            # assume boxes = [[xmin, ymin, xmax, ymax, ...
            #   might want to split by class because we could have
            #   a car inside the bounding box of a plane, for example
            boxes_nms_input = np.stack(
                (xmins, ymins, xmaxs, ymaxs), axis=1)
            if use_weighted_nms:
                probs = scores
            else:
                probs = []
            good_idxs = non_max_suppression(
                boxes_nms_input, probs=probs,
                overlapThresh=nms_overlap_thresh)

            if verbose:
                print("num boxes_all:", len(xmins))
                print("num good_idxs:", len(good_idxs))
            boxes = boxes[good_idxs]
            scores = scores[good_idxs]
            df_idxs = df_idxs[good_idxs]

            df_idxs_tot.extend(df_idxs)
            count += len(df_idxs)

    df_idxs_tot_final = np.unique(df_idxs_tot)

    # create dataframe
    if verbose:
        print("df idxs::", df.index)
        print("df_idxs_tot_final:", df_idxs_tot_final)
    df_out = df.loc[df_idxs_tot_final]

    t1 = time.time()
    print("Initial length:", len(df), "Final length:", len(df_out))
    print("Time to run refine_df():", t1-t0, "seconds")
    return df_out  