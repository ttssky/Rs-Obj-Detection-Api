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

__all__ = ["add_geo_coords_to_df"]

def add_geo_coords_to_df(df_, Proj_str='epsg:3857',
                         create_geojson=True, verbose=False):
    """
    将预测框转化为Geojson

    Arguments
    ---------
    df_ : pandas dataframe
        预测结果的dataframe
    Proj_str : 
        生成geojson的坐标系，默认3857
    create_geojson : boolean
        是否保存为geojson,默认True
    verbose : boolean
        是否print

    Returns
    -------
    df_, json : tuple
        df_ 更新后的dataframe，包括投影坐标和地理坐标
        json 预测框转化的geojson
    """

    t0 = time.time()
    print("Adding geo coords...")

    out_arr_json = []
    out_arr = []
    for idx, row in df_.iterrows():

        out_arr_row, poly_geo = get_row_geo_coords(
            row, 
            Proj_str=Proj_str,
            verbose=verbose)
        out_arr.append(out_arr_row)
        if create_geojson:
            out_arr_json.append(poly_geo)

    # 更新dataframe
    # [lon0, lat0, lon1, lat1, x0_wmp, y0_wmp, x1_wmp, y1_wmp]
    out_arr = np.array(out_arr)
    df_['lon0'] = out_arr[:, 0]
    df_['lat0'] = out_arr[:, 1]
    df_['lon1'] = out_arr[:, 2]
    df_['lat1'] = out_arr[:, 3]
    df_['x0_wmp'] = out_arr[:, 4]
    df_['y0_wmp'] = out_arr[:, 5]
    df_['x1_wmp'] = out_arr[:, 6]
    df_['y1_wmp'] = out_arr[:, 7]

    # 创建geodataframe
    #   https://gis.stackexchange.com/questions/174159/convert-a-pandas-dataframe-to-a-geodataframe
    if create_geojson and (len(out_arr_json) > 0):
        crs_init = {'init': Proj_str}
        df_json = pd.DataFrame(out_arr_json, columns=['geometry'])
        # 为geodataframe创建必要的字段
        df_json['category'] = df_['Category'].values
        df_json['prob'] = df_['Prob'].values
        gdf = gpd.GeoDataFrame(df_json, crs=crs_init, geometry=out_arr_json)
        json_out = gdf
        # json_out = gdf.to_json()
    else:
        json_out = []

    t1 = time.time()
    print("Time to add geo coords to df:", t1 - t0, "seconds")
    return df_, json_out

def get_row_geo_coords(row, 
                       Proj_str='epsg:3857',
                       verbose=False):

    """
    将dataframe的每一行转化为geojson.

    Arguments
    ---------
    row : pandas dataframe
        dataframe的每一行
    Proj_str : str
        坐标系默认3857
    verbose : boolean
        是否print
    Returns
    -------
    out_arr, poly_geo : tuple
        out_arr是每个目标框的投影坐标与地理坐标
        [lon0, lat0, lon1, lat1, x0_wmp, y0_wmp, x1_wmp, y1_wmp]
        ploy_geo是有坐标的shapely对象
    """

    # convert latlon to wmp

    x0, y0 = row['Xmin_Glob'], row['Ymin_Glob']
    x1, y1 = row['Xmax_Glob'], row['Ymax_Glob']
    poly_geo = shapely.geometry.Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
    inProj = pyproj.Proj(init=Proj_str)

    if Proj_str.lower() == 'epsg:3857':
        outProj = pyproj.Proj(init='epsg:4326')
        x0_wmp, y0_wmp, x1_wmp, y1_wmp = poly_geo.bounds  
        lon0, lat0 = pyproj.transform(inProj, outProj, x0_wmp, y0_wmp)
        lon1, lat1 = pyproj.transform(inProj, outProj, x1_wmp, y1_wmp)
    else:
        outProj = pyproj.Proj(init='epsg:3857')
        lon0, lat0, lon1, lat1 = poly_geo.bounds  
        x0_wmp, y0_wmp = pyproj.transform(inProj, outProj, lon0, lat0)
        x1_wmp, y1_wmp = pyproj.transform(inProj, outProj, lon1, lat1)
      
    if verbose:
        print("idx,  x0, y0, x1, y1:", row.values[0], x0, y0, x1, y1)


    if verbose:
        print("  lon0, lat0, lon1, lat1:", lon0, lat0, lon1, lat1)


    # 返回目标框的两种坐标
    out_arr = [lon0, lat0, lon1, lat1, x0_wmp, y0_wmp, x1_wmp, y1_wmp]

    return out_arr, poly_geo
