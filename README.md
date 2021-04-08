# ![图片](https://uploader.shimo.im/f/lkSfaFExEShNv8EV.png!thumbnail?fileGuid=yx6Ck3yggDgQpDWW)

1. 实时目标检测与分割
## 1.1 安装

A. 安装gdal

参考[https://stackoverflow.com/questions/37294127/python-gdal-2-1-installation-on-ubuntu-16-04](https://stackoverflow.com/questions/37294127/python-gdal-2-1-installation-on-ubuntu-16-04?fileGuid=yx6Ck3yggDgQpDWW)

B. 安装Python依赖

```plain
~ pip install -r requirements
```
C. heycloud服务heycoud-data-api
D. 可以直接使用geohey@192.168.31.14机器/data2/lvyikai_proj/p01_htwy_proj/Rs-Obj-Detection-Api中的工程，环境已经部署好。

## 1.2 工程目录结构

```json
Rs-Obj-Detection-Api
├─ app
│  ├─ __init__.py
│  ├─ detection #目标检测接口
│  │  ├─ __init__.py
│  │  ├─ core
│  │  │  ├─ __init__.py
│  │  │  ├─ add_georeference.py #生成geojson保存返回
│  │  │  ├─ non_max_suppression.py #极大值抑制nms
│  │  │  ├─ post_process.py #预测框添加空间坐标
│  │  ├─ darknet.py #python与darknet建立bindings
│  │  ├─ utils.py
│  │  └─ views.py #定义目标检测视图函数
│  ├─ models.py #orm模型，调用数据库影像元数据
│  ├─ request.py #多线程从heycloud请求瓦片
│  └─ segmentation #地物分割接口
│     ├─ __init__.py
│     ├─ segmentation_models_pytorch #pytorch分割模型
│     │  ├─ __version__.py
│     │  ├─ base #模型基类
│     │  ├─ encoders #特征提取器
│     │  └─ fpn #模型框架(方法)
│     ├─ utils.py
│     └─ views.py
├─ config.py #配置文件
├─ darknet #darknet深度学习框架
│  ├─ Makefile
│  ├─ README.md
│  ├─ darknet
│  ├─ obj
│  ├─ src
└─ run.py #debug模式开启服务
└─ requirements.txt
```
## 1.3 配置文件说明（config.py）

```python
import os
basedir = os.path.abspath(os.path.dirname(__file__))
image_request_heycloud_host = 'http://192.168.31.17' #获取预测影像的heycloud-data-api地址
port = 9000 #获取预测影像的heycloud-data-api的端口号
#存储影像信息的sqlalchemy格式pg地址
db_sqlalchemy_addr = 'postgresql+psycopg2://postgres:postgres@192.168.31.17:15432/heycloud'
#目标检测预测模型信息
det_method = 'yolov4' #目标检测方法
det_model_ver = 'v0.0.1'
det_name_prefix = det_method + '-' + det_model_ver
dl_detection_dir = '/home/geohey/volumes/dl-detection/' 
#语义分割模型信息
seg_method = 'efficientnet-b4' #语义分割方法
seg_model_ver = 'v0.0.1'
seg_name_prefix = seg_method + '-' + seg_model_ver
dl_segmentation_dir = '/home/geohey/volumes/dl-segmentation/'
#Flask app 配置 
class Config:
    IMAGE_REQUEST_HOST = image_request_heycloud_host + ':' + str(port)
    IMAGE_REQUEST_HEADERS = {
        "x-heycloud-admin-session": "tFn8aWIgWHnYTCO/NR/r2OK4wef96gtC",
        "Content-Type":"application/json"
    }
    @staticmethod
    def init_app(app):
        pass
```
* 上述配置是默认heycloud部署与17机器上
* 模型文件与预测结果以如下目录组织：
    * 模型文件路径：/home/geohey/volumes/dl-detection/{category}/{location}/weights/{model_version}/*.weights
    * 预测结果路径：/home/geohey/volumes/dl-detection/{category}/{location}/weights/{model_version}/files/{uid}/results/results.geojson
## 1.4 飞机目标检测接口

### ① 调用形式

```plain
http://host:port/detection/{category}
```
* category表示类别，可选：aircraft
### ② Post参数

```json
{
  "bbox": [15522691.7630018796771765,4225193.8223429527133703, 15524264.7937226444482803,4227109.6729318778961897],
  "idatasetId": "2b7c25a6-3c5d-4653-bb08-285086247f01",
  "location": "atsugi",
  "overlap": 0.3,
  "conf_thresh": 0.25,
  "height": 608,
  "width": 608
}
```
### ③ 参数说明

* bbox：待预测空间范围（3857）
* idatasetId：heycloud中影像数据集的id
* location：待预测基地名称，可选：atsugi（厚木）、andersen（安德森）、iwakuni（岩国）、misawa（三泽）、kadena（嘉手纳）
* overlap：(0, 1]，默认0.3
* heights：裁剪图片高，默认608
* width：裁剪图片宽，默认608
* 注：飞机检测模型是按照长宽608的图片进行训练，预测最好使用默认608
## 1.5 基地建筑分割

### ① 调用形式

```plain
http://host:port/segmentation/{category}
```
* category表示类别，可选：buildings
### ② Post参数

```json
{
  "bbox": [15522563.7544428743422031,4224912.9063391108065844, 15524244.0752789303660393,4227099.8194252224639058],
  "idatasetId": "2b7c25a6-3c5d-4653-bb08-285086247f01",
  "location": "atsugi",
  "overlap": 0.3,
  "heights": 512,
  "width": 512
}
```
### ③ 参数说明

* bbox：待预测空间范围（3857）
* idatasetId：heycloud中影像数据集的id
* location：待预测基地名称，可选：atsugi（厚木）、andersen（安德森）、iwakuni（岩国）、misawa（三泽）、kadena（嘉手纳）
* overlap：(0, 1]，默认0.3
* heights：裁剪图片高，默认512
* width：裁剪图片宽，默认512
* 注：基地建筑分割模型是按照长宽512的图片进行训练，预测最好使用默认512
2. 目标检测-预测过程
## 2.1 heycloud获取瓦片

* app/request.py 代码实现

获取bbox范围后，预测接口会按照长宽参数从heycloud-data-api中多线程并发获取瓦片，并且瓦片的名称以如下格式存储：

```plain
image__{左上角x}_{左上角y}_{瓦片height}_{瓦片width}_{影像height}_{影像width}.png
```
例
```plain
image__1526230.9862189812_16135005.754588803_512_512_19648_16288.png
```
* 表示此瓦片左上角坐标（1526230.9862189812_16135005.754588803），height和width为512，所在影像height和width为19648和16288。
* 每一次请求，瓦片存储在以下路径下：
```plain
/home/geohey/volumes/dl-detection/{category}/{location}/weights/{model_version}/files/{uid}/images
```
## 2.2 每张瓦片进行预测

* app/detection/darknet.py 代码实现（建立python darknet bindings）
* 根据请求参数中的预测地物类型和预测基地名称，寻找对应路径下的模型，并自动生成预测所需要的inference.data和瓦片路径images_path.txt文件。
* 每张瓦片输入darknet进行预测，预测结果保存在：
```plain
/home/geohey/volumes/dl-detection/{category}/{location}/weights/{model_version}/files/{uid}/results/darknet_infer_res/类别名.txt
```
* 类别名.txt中，存储了预测后含有目标体的瓦片路径，置信度，以及bbox坐标（瓦片的相对坐标）
## 2.3 后处理

### 2.3.1 添加空间坐标

* app/detection/core/post_process.py 代码实现
* 根据darknet预测的txt文件和对应的瓦片名称以及预测框的信息，生成pandas的dataframe
* 根据预测框距离边缘的距离以及预测框的长宽比，过滤掉一部分预测框。（距离边缘近的或者长宽比大的预测框一般是裁剪过程中被一分为二的物体）
* 根据瓦片名称记录的左上角空间坐标信息和txt中的相对坐标信息，计算每个预测框的全局坐标信息，得到真实的空间坐标。记录在pandas的dataframe中。这个dataframe以csv格式保存在：
```plain
/home/geohey/volumes/dl-detection/{category}/{location}/weights/{model_version}/files/{uid}/results/results.csv
```
### 2.3.2 非极大值抑制（NMS）

* app/detection/core/non_max_suppression.py 代码实现
* 由于yolo中有多个anchor负责预测不同尺度的目标框，那么同一个物体可能被多个anchor检测到，导致输出多个框。
* nms就是提取置信度高的目标检测框，而抑制置信度低的误检框的一种算法，基本的nms思路比较简单。参考[https://mp.weixin.qq.com/s/orYMdwZ1VwwIScPmIiq5iA](https://mp.weixin.qq.com/s/orYMdwZ1VwwIScPmIiq5iA?fileGuid=yx6Ck3yggDgQpDWW)
* 首先根据请求参数中的置信度阈值进行筛选（大于此值的目标框保留），然后根据请求参数中的nms阈值和dataframe中的目标框全局坐标，进行整幅影像的全局nms（常规nms只对单张瓦片进行nms）
### 2.3.3 生成geojson

* 根据最终后处理完成后和目标框，利用geopandas中的geodataframe生成geojson，保存在：
```plain
/home/geohey/volumes/dl-detection/{category}/{location}/weights/{model_version}/files/{uid}/results/results.geojson
```
3. 建筑分割-预测过程

heycloud获取瓦片----> 预测单张瓦片 ----> 根据原始坐标赋予预测瓦片坐标 ---> gdal_merge成整张预测图 ----> polygonsize ----> 返回

