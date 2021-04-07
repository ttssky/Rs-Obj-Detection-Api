# Rs-Obj-Detection-Api
以flask作为web框架，darknet作为后端深度学习推断框架，实现卫星影像的目标检测API


```
Rs-Obj-Detection-Api
├─ app
│  ├─ __init__.py
│  ├─ detection
│  │  ├─ __init__.py
│  │  ├─ core
│  │  │  ├─ __init__.py
│  │  │  ├─ add_georeference.py
│  │  │  ├─ non_max_suppression.py
│  │  │  ├─ post_process.py
│  │  │  └─ slice_ims.py
│  │  ├─ darknet.py
│  │  ├─ errors.py
│  │  ├─ utils.py
│  │  └─ views.py
│  ├─ models.py
│  ├─ request.py
│  └─ segmentation
│     ├─ __init__.py
│     ├─ segmentation_models_pytorch
│     │  ├─ __version__.py
│     │  ├─ base
│     │  ├─ encoders
│     │  └─ fpn
│     ├─ utils.py
│     └─ views.py
├─ config.py
├─ darknet
│  ├─ Makefile
│  ├─ README.md
│  ├─ darknet
│  ├─ obj
│  ├─ src
└─ run.py

```
```
Rs-Obj-Detection-Api
├─ .vscode
│  └─ settings.json
├─ README.md
├─ __pycache__
│  └─ config.cpython-36.pyc
├─ app
│  ├─ __init__.py
│  ├─ __pycache__
│  │  ├─ __init__.cpython-36.pyc
│  │  ├─ models.cpython-36.pyc
│  │  └─ request.cpython-36.pyc
│  ├─ detection
│  │  ├─ __init__.py
│  │  ├─ __pycache__
│  │  │  ├─ __init__.cpython-36.pyc
│  │  │  ├─ config.cpython-36.pyc
│  │  │  ├─ darknet.cpython-36.pyc
│  │  │  ├─ errors.cpython-36.pyc
│  │  │  ├─ request.cpython-36.pyc
│  │  │  ├─ requests.cpython-36.pyc
│  │  │  ├─ utils.cpython-36.pyc
│  │  │  └─ views.cpython-36.pyc
│  │  ├─ core
│  │  │  ├─ __init__.py
│  │  │  ├─ __pycache__
│  │  │  │  ├─ __init__.cpython-36.pyc
│  │  │  │  ├─ add_georeference.cpython-36.pyc
│  │  │  │  ├─ non_max_suppression.cpython-36.pyc
│  │  │  │  ├─ post_process.cpython-36.pyc
│  │  │  │  └─ slice_ims.cpython-36.pyc
│  │  │  ├─ add_georeference.py
│  │  │  ├─ non_max_suppression.py
│  │  │  ├─ post_process.py
│  │  │  └─ slice_ims.py
│  │  ├─ darknet.py
│  │  ├─ errors.py
│  │  ├─ utils.py
│  │  └─ views.py
│  ├─ models.py
│  ├─ request.py
│  └─ segmentation
│     ├─ __init__.py
│     ├─ __pycache__
│     │  ├─ __init__.cpython-36.pyc
│     │  ├─ utils.cpython-36.pyc
│     │  └─ views.cpython-36.pyc
│     ├─ segmentation_models_pytorch
│     │  ├─ __init__.py
│     │  ├─ __pycache__
│     │  │  ├─ __init__.cpython-36.pyc
│     │  │  ├─ __init__.cpython-37.pyc
│     │  │  └─ __version__.cpython-36.pyc
│     │  ├─ __version__.py
│     │  ├─ base
│     │  │  ├─ __init__.py
│     │  │  ├─ __pycache__
│     │  │  │  ├─ __init__.cpython-36.pyc
│     │  │  │  ├─ heads.cpython-36.pyc
│     │  │  │  ├─ initialization.cpython-36.pyc
│     │  │  │  ├─ model.cpython-36.pyc
│     │  │  │  └─ modules.cpython-36.pyc
│     │  │  ├─ heads.py
│     │  │  ├─ initialization.py
│     │  │  ├─ model.py
│     │  │  └─ modules.py
│     │  ├─ encoders
│     │  │  ├─ _EfficientNet-PyTorch
│     │  │  │  ├─ LICENSE
│     │  │  │  ├─ README.md
│     │  │  │  ├─ examples
│     │  │  │  │  ├─ imagenet
│     │  │  │  │  │  ├─ README.md
│     │  │  │  │  │  ├─ data
│     │  │  │  │  │  │  └─ README.md
│     │  │  │  │  │  └─ main.py
│     │  │  │  │  └─ simple
│     │  │  │  │     ├─ check.ipynb
│     │  │  │  │     ├─ example.ipynb
│     │  │  │  │     ├─ img.jpg
│     │  │  │  │     ├─ img2.jpg
│     │  │  │  │     └─ labels_map.txt
│     │  │  │  ├─ hubconf.py
│     │  │  │  ├─ setup.py
│     │  │  │  ├─ tests
│     │  │  │  │  └─ test_model.py
│     │  │  │  └─ tf_to_pytorch
│     │  │  │     ├─ README.md
│     │  │  │     ├─ convert_tf_to_pt
│     │  │  │     │  ├─ download.sh
│     │  │  │     │  ├─ load_tf_weights.py
│     │  │  │     │  ├─ load_tf_weights_tf1.py
│     │  │  │     │  ├─ original_tf
│     │  │  │     │  │  ├─ __init__.py
│     │  │  │     │  │  ├─ efficientnet_builder.py
│     │  │  │     │  │  ├─ efficientnet_model.py
│     │  │  │     │  │  ├─ eval_ckpt_main.py
│     │  │  │     │  │  ├─ eval_ckpt_main_tf1.py
│     │  │  │     │  │  ├─ preprocessing.py
│     │  │  │     │  │  └─ utils.py
│     │  │  │     │  ├─ rename.sh
│     │  │  │     │  └─ run.sh
│     │  │  │     └─ pretrained_tensorflow
│     │  │  │        └─ download.sh
│     │  │  ├─ __init__.py
│     │  │  ├─ __pycache__
│     │  │  │  ├─ __init__.cpython-36.pyc
│     │  │  │  ├─ _base.cpython-36.pyc
│     │  │  │  ├─ _preprocessing.cpython-36.pyc
│     │  │  │  ├─ _utils.cpython-36.pyc
│     │  │  │  ├─ densenet.cpython-36.pyc
│     │  │  │  ├─ dpn.cpython-36.pyc
│     │  │  │  ├─ efficientnet.cpython-36.pyc
│     │  │  │  ├─ inceptionresnetv2.cpython-36.pyc
│     │  │  │  ├─ inceptionv4.cpython-36.pyc
│     │  │  │  ├─ mobilenet.cpython-36.pyc
│     │  │  │  ├─ resnet.cpython-36.pyc
│     │  │  │  ├─ senet.cpython-36.pyc
│     │  │  │  ├─ vgg.cpython-36.pyc
│     │  │  │  └─ xception.cpython-36.pyc
│     │  │  ├─ _base.py
│     │  │  ├─ _preprocessing.py
│     │  │  ├─ _utils.py
│     │  │  ├─ efficientnet.py
│     │  │  └─ efficientnet_pytorch
│     │  │     ├─ __init__.py
│     │  │     ├─ __pycache__
│     │  │     │  ├─ __init__.cpython-36.pyc
│     │  │     │  ├─ model.cpython-36.pyc
│     │  │     │  └─ utils.cpython-36.pyc
│     │  │     ├─ model.py
│     │  │     └─ utils.py
│     │  └─ fpn
│     │     ├─ __init__.py
│     │     ├─ __pycache__
│     │     │  ├─ __init__.cpython-36.pyc
│     │     │  ├─ decoder.cpython-36.pyc
│     │     │  └─ model.cpython-36.pyc
│     │     ├─ decoder.py
│     │     └─ model.py
│     ├─ utils.py
│     └─ views.py
├─ config.py
├─ darknet
│  ├─ 3rdparty
│  │  ├─ pthreads
│  │  │  ├─ bin
│  │  │  │  ├─ pthreadGC2.dll
│  │  │  │  └─ pthreadVC2.dll
│  │  │  ├─ include
│  │  │  │  ├─ pthread.h
│  │  │  │  ├─ sched.h
│  │  │  │  └─ semaphore.h
│  │  │  └─ lib
│  │  │     ├─ libpthreadGC2.a
│  │  │     └─ pthreadVC2.lib
│  │  └─ stb
│  │     └─ include
│  │        ├─ stb_image.h
│  │        └─ stb_image_write.h
│  ├─ CMakeLists.txt
│  ├─ DarknetConfig.cmake.in
│  ├─ LICENSE
│  ├─ Makefile
│  ├─ README.md
│  ├─ backup
│  ├─ build
│  │  └─ darknet
│  │     ├─ YoloWrapper.cs
│  │     ├─ darknet.sln
│  │     ├─ darknet.vcxproj
│  │     ├─ darknet_no_gpu.sln
│  │     ├─ darknet_no_gpu.vcxproj
│  │     ├─ x64
│  │     │  ├─ calc_anchors.cmd
│  │     │  ├─ calc_mAP.cmd
│  │     │  ├─ calc_mAP_coco.cmd
│  │     │  ├─ calc_mAP_voc_py.cmd
│  │     │  ├─ cfg
│  │     │  │  ├─ Gaussian_yolov3_BDD.cfg
│  │     │  │  ├─ alexnet.cfg
│  │     │  │  ├─ cd53paspp-gamma.cfg
│  │     │  │  ├─ cifar.cfg
│  │     │  │  ├─ cifar.test.cfg
│  │     │  │  ├─ coco.data
│  │     │  │  ├─ combine9k.data
│  │     │  │  ├─ crnn.train.cfg
│  │     │  │  ├─ csdarknet53-omega.cfg
│  │     │  │  ├─ cspx-p7-mish-omega.cfg
│  │     │  │  ├─ cspx-p7-mish_hp.cfg
│  │     │  │  ├─ csresnext50-panet-spp-original-optimal.cfg
│  │     │  │  ├─ csresnext50-panet-spp.cfg
│  │     │  │  ├─ darknet.cfg
│  │     │  │  ├─ darknet19.cfg
│  │     │  │  ├─ darknet19_448.cfg
│  │     │  │  ├─ darknet53.cfg
│  │     │  │  ├─ darknet53_448_xnor.cfg
│  │     │  │  ├─ densenet201.cfg
│  │     │  │  ├─ efficientnet-lite3.cfg
│  │     │  │  ├─ efficientnet_b0.cfg
│  │     │  │  ├─ enet-coco.cfg
│  │     │  │  ├─ extraction.cfg
│  │     │  │  ├─ extraction.conv.cfg
│  │     │  │  ├─ extraction22k.cfg
│  │     │  │  ├─ go.test.cfg
│  │     │  │  ├─ gru.cfg
│  │     │  │  ├─ imagenet1k.data
│  │     │  │  ├─ imagenet22k.dataset
│  │     │  │  ├─ imagenet9k.hierarchy.dataset
│  │     │  │  ├─ jnet-conv.cfg
│  │     │  │  ├─ lstm.train.cfg
│  │     │  │  ├─ openimages.data
│  │     │  │  ├─ resnet101.cfg
│  │     │  │  ├─ resnet152.cfg
│  │     │  │  ├─ resnet152_trident.cfg
│  │     │  │  ├─ resnet50.cfg
│  │     │  │  ├─ resnext152-32x4d.cfg
│  │     │  │  ├─ rnn.cfg
│  │     │  │  ├─ rnn.train.cfg
│  │     │  │  ├─ strided.cfg
│  │     │  │  ├─ t1.test.cfg
│  │     │  │  ├─ tiny-yolo-voc.cfg
│  │     │  │  ├─ tiny-yolo.cfg
│  │     │  │  ├─ tiny-yolo_xnor.cfg
│  │     │  │  ├─ tiny.cfg
│  │     │  │  ├─ vgg-16.cfg
│  │     │  │  ├─ vgg-conv.cfg
│  │     │  │  ├─ voc.data
│  │     │  │  ├─ writing.cfg
│  │     │  │  ├─ yolo-voc.2.0.cfg
│  │     │  │  ├─ yolo-voc.cfg
│  │     │  │  ├─ yolo.2.0.cfg
│  │     │  │  ├─ yolo.cfg
│  │     │  │  ├─ yolo9000.cfg
│  │     │  │  ├─ yolov2-tiny-voc.cfg
│  │     │  │  ├─ yolov2-tiny.cfg
│  │     │  │  ├─ yolov2-voc.cfg
│  │     │  │  ├─ yolov2.cfg
│  │     │  │  ├─ yolov3-openimages.cfg
│  │     │  │  ├─ yolov3-spp.cfg
│  │     │  │  ├─ yolov3-tiny-prn.cfg
│  │     │  │  ├─ yolov3-tiny.cfg
│  │     │  │  ├─ yolov3-tiny_3l.cfg
│  │     │  │  ├─ yolov3-tiny_obj.cfg
│  │     │  │  ├─ yolov3-tiny_occlusion_track.cfg
│  │     │  │  ├─ yolov3-tiny_xnor.cfg
│  │     │  │  ├─ yolov3-voc.cfg
│  │     │  │  ├─ yolov3-voc.yolov3-giou-40.cfg
│  │     │  │  ├─ yolov3.cfg
│  │     │  │  ├─ yolov3.coco-giou-12.cfg
│  │     │  │  ├─ yolov3_5l.cfg
│  │     │  │  ├─ yolov4-custom.cfg
│  │     │  │  ├─ yolov4-tiny-3l.cfg
│  │     │  │  ├─ yolov4-tiny-custom.cfg
│  │     │  │  ├─ yolov4-tiny.cfg
│  │     │  │  ├─ yolov4-tiny_contrastive.cfg
│  │     │  │  └─ yolov4.cfg
│  │     │  ├─ classifier_densenet201.cmd
│  │     │  ├─ classifier_resnet50.cmd
│  │     │  ├─ darknet.py
│  │     │  ├─ darknet_coco.cmd
│  │     │  ├─ darknet_coco_9000.cmd
│  │     │  ├─ darknet_coco_9000_demo.cmd
│  │     │  ├─ darknet_demo_coco.cmd
│  │     │  ├─ darknet_demo_json_stream.cmd
│  │     │  ├─ darknet_demo_mjpeg_stream.cmd
│  │     │  ├─ darknet_demo_store.cmd
│  │     │  ├─ darknet_demo_voc.cmd
│  │     │  ├─ darknet_demo_voc_param.cmd
│  │     │  ├─ darknet_demo_voc_tiny.cmd
│  │     │  ├─ darknet_json_reslut.cmd
│  │     │  ├─ darknet_many_images.cmd
│  │     │  ├─ darknet_net_cam_coco.cmd
│  │     │  ├─ darknet_net_cam_voc.cmd
│  │     │  ├─ darknet_python.cmd
│  │     │  ├─ darknet_tiny_v2.cmd
│  │     │  ├─ darknet_video.cmd
│  │     │  ├─ darknet_video.py
│  │     │  ├─ darknet_voc.cmd
│  │     │  ├─ darknet_voc_tiny_v2.cmd
│  │     │  ├─ darknet_web_cam_voc.cmd
│  │     │  ├─ darknet_yolo_v3.cmd
│  │     │  ├─ darknet_yolo_v3_openimages.cmd
│  │     │  ├─ darknet_yolo_v3_video.cmd
│  │     │  ├─ darknet_yolov3_pseudo_labeling.cmd
│  │     │  ├─ data
│  │     │  │  ├─ 9k.labels
│  │     │  │  ├─ 9k.names
│  │     │  │  ├─ 9k.tree
│  │     │  │  ├─ coco.data
│  │     │  │  ├─ coco.names
│  │     │  │  ├─ coco9k.map
│  │     │  │  ├─ combine9k.data
│  │     │  │  ├─ dog.jpg
│  │     │  │  ├─ eagle.jpg
│  │     │  │  ├─ giraffe.jpg
│  │     │  │  ├─ goal.txt
│  │     │  │  ├─ horses.jpg
│  │     │  │  ├─ imagenet.labels.list
│  │     │  │  ├─ imagenet.shortnames.list
│  │     │  │  ├─ inet9k.map
│  │     │  │  ├─ labels
│  │     │  │  │  ├─ 100_0.png
│  │     │  │  │  ├─ 100_1.png
│  │     │  │  │  ├─ 100_2.png
│  │     │  │  │  ├─ 100_3.png
│  │     │  │  │  ├─ 100_4.png
│  │     │  │  │  ├─ 100_5.png
│  │     │  │  │  ├─ 100_6.png
│  │     │  │  │  ├─ 100_7.png
│  │     │  │  │  ├─ 101_0.png
│  │     │  │  │  ├─ 101_1.png
│  │     │  │  │  ├─ 101_2.png
│  │     │  │  │  ├─ 101_3.png
│  │     │  │  │  ├─ 101_4.png
│  │     │  │  │  ├─ 101_5.png
│  │     │  │  │  ├─ 101_6.png
│  │     │  │  │  ├─ 101_7.png
│  │     │  │  │  ├─ 102_0.png
│  │     │  │  │  ├─ 102_1.png
│  │     │  │  │  ├─ 102_2.png
│  │     │  │  │  ├─ 102_3.png
│  │     │  │  │  ├─ 102_4.png
│  │     │  │  │  ├─ 102_5.png
│  │     │  │  │  ├─ 102_6.png
│  │     │  │  │  ├─ 102_7.png
│  │     │  │  │  ├─ 103_0.png
│  │     │  │  │  ├─ 103_1.png
│  │     │  │  │  ├─ 103_2.png
│  │     │  │  │  ├─ 103_3.png
│  │     │  │  │  ├─ 103_4.png
│  │     │  │  │  ├─ 103_5.png
│  │     │  │  │  ├─ 103_6.png
│  │     │  │  │  ├─ 103_7.png
│  │     │  │  │  ├─ 104_0.png
│  │     │  │  │  ├─ 104_1.png
│  │     │  │  │  ├─ 104_2.png
│  │     │  │  │  ├─ 104_3.png
│  │     │  │  │  ├─ 104_4.png
│  │     │  │  │  ├─ 104_5.png
│  │     │  │  │  ├─ 104_6.png
│  │     │  │  │  ├─ 104_7.png
│  │     │  │  │  ├─ 105_0.png
│  │     │  │  │  ├─ 105_1.png
│  │     │  │  │  ├─ 105_2.png
│  │     │  │  │  ├─ 105_3.png
│  │     │  │  │  ├─ 105_4.png
│  │     │  │  │  ├─ 105_5.png
│  │     │  │  │  ├─ 105_6.png
│  │     │  │  │  ├─ 105_7.png
│  │     │  │  │  ├─ 106_0.png
│  │     │  │  │  ├─ 106_1.png
│  │     │  │  │  ├─ 106_2.png
│  │     │  │  │  ├─ 106_3.png
│  │     │  │  │  ├─ 106_4.png
│  │     │  │  │  ├─ 106_5.png
│  │     │  │  │  ├─ 106_6.png
│  │     │  │  │  ├─ 106_7.png
│  │     │  │  │  ├─ 107_0.png
│  │     │  │  │  ├─ 107_1.png
│  │     │  │  │  ├─ 107_2.png
│  │     │  │  │  ├─ 107_3.png
│  │     │  │  │  ├─ 107_4.png
│  │     │  │  │  ├─ 107_5.png
│  │     │  │  │  ├─ 107_6.png
│  │     │  │  │  ├─ 107_7.png
│  │     │  │  │  ├─ 108_0.png
│  │     │  │  │  ├─ 108_1.png
│  │     │  │  │  ├─ 108_2.png
│  │     │  │  │  ├─ 108_3.png
│  │     │  │  │  ├─ 108_4.png
│  │     │  │  │  ├─ 108_5.png
│  │     │  │  │  ├─ 108_6.png
│  │     │  │  │  ├─ 108_7.png
│  │     │  │  │  ├─ 109_0.png
│  │     │  │  │  ├─ 109_1.png
│  │     │  │  │  ├─ 109_2.png
│  │     │  │  │  ├─ 109_3.png
│  │     │  │  │  ├─ 109_4.png
│  │     │  │  │  ├─ 109_5.png
│  │     │  │  │  ├─ 109_6.png
│  │     │  │  │  ├─ 109_7.png
│  │     │  │  │  ├─ 110_0.png
│  │     │  │  │  ├─ 110_1.png
│  │     │  │  │  ├─ 110_2.png
│  │     │  │  │  ├─ 110_3.png
│  │     │  │  │  ├─ 110_4.png
│  │     │  │  │  ├─ 110_5.png
│  │     │  │  │  ├─ 110_6.png
│  │     │  │  │  ├─ 110_7.png
│  │     │  │  │  ├─ 111_0.png
│  │     │  │  │  ├─ 111_1.png
│  │     │  │  │  ├─ 111_2.png
│  │     │  │  │  ├─ 111_3.png
│  │     │  │  │  ├─ 111_4.png
│  │     │  │  │  ├─ 111_5.png
│  │     │  │  │  ├─ 111_6.png
│  │     │  │  │  ├─ 111_7.png
│  │     │  │  │  ├─ 112_0.png
│  │     │  │  │  ├─ 112_1.png
│  │     │  │  │  ├─ 112_2.png
│  │     │  │  │  ├─ 112_3.png
│  │     │  │  │  ├─ 112_4.png
│  │     │  │  │  ├─ 112_5.png
│  │     │  │  │  ├─ 112_6.png
│  │     │  │  │  ├─ 112_7.png
│  │     │  │  │  ├─ 113_0.png
│  │     │  │  │  ├─ 113_1.png
│  │     │  │  │  ├─ 113_2.png
│  │     │  │  │  ├─ 113_3.png
│  │     │  │  │  ├─ 113_4.png
│  │     │  │  │  ├─ 113_5.png
│  │     │  │  │  ├─ 113_6.png
│  │     │  │  │  ├─ 113_7.png
│  │     │  │  │  ├─ 114_0.png
│  │     │  │  │  ├─ 114_1.png
│  │     │  │  │  ├─ 114_2.png
│  │     │  │  │  ├─ 114_3.png
│  │     │  │  │  ├─ 114_4.png
│  │     │  │  │  ├─ 114_5.png
│  │     │  │  │  ├─ 114_6.png
│  │     │  │  │  ├─ 114_7.png
│  │     │  │  │  ├─ 115_0.png
│  │     │  │  │  ├─ 115_1.png
│  │     │  │  │  ├─ 115_2.png
│  │     │  │  │  ├─ 115_3.png
│  │     │  │  │  ├─ 115_4.png
│  │     │  │  │  ├─ 115_5.png
│  │     │  │  │  ├─ 115_6.png
│  │     │  │  │  ├─ 115_7.png
│  │     │  │  │  ├─ 116_0.png
│  │     │  │  │  ├─ 116_1.png
│  │     │  │  │  ├─ 116_2.png
│  │     │  │  │  ├─ 116_3.png
│  │     │  │  │  ├─ 116_4.png
│  │     │  │  │  ├─ 116_5.png
│  │     │  │  │  ├─ 116_6.png
│  │     │  │  │  ├─ 116_7.png
│  │     │  │  │  ├─ 117_0.png
│  │     │  │  │  ├─ 117_1.png
│  │     │  │  │  ├─ 117_2.png
│  │     │  │  │  ├─ 117_3.png
│  │     │  │  │  ├─ 117_4.png
│  │     │  │  │  ├─ 117_5.png
│  │     │  │  │  ├─ 117_6.png
│  │     │  │  │  ├─ 117_7.png
│  │     │  │  │  ├─ 118_0.png
│  │     │  │  │  ├─ 118_1.png
│  │     │  │  │  ├─ 118_2.png
│  │     │  │  │  ├─ 118_3.png
│  │     │  │  │  ├─ 118_4.png
│  │     │  │  │  ├─ 118_5.png
│  │     │  │  │  ├─ 118_6.png
│  │     │  │  │  ├─ 118_7.png
│  │     │  │  │  ├─ 119_0.png
│  │     │  │  │  ├─ 119_1.png
│  │     │  │  │  ├─ 119_2.png
│  │     │  │  │  ├─ 119_3.png
│  │     │  │  │  ├─ 119_4.png
│  │     │  │  │  ├─ 119_5.png
│  │     │  │  │  ├─ 119_6.png
│  │     │  │  │  ├─ 119_7.png
│  │     │  │  │  ├─ 120_0.png
│  │     │  │  │  ├─ 120_1.png
│  │     │  │  │  ├─ 120_2.png
│  │     │  │  │  ├─ 120_3.png
│  │     │  │  │  ├─ 120_4.png
│  │     │  │  │  ├─ 120_5.png
│  │     │  │  │  ├─ 120_6.png
│  │     │  │  │  ├─ 120_7.png
│  │     │  │  │  ├─ 121_0.png
│  │     │  │  │  ├─ 121_1.png
│  │     │  │  │  ├─ 121_2.png
│  │     │  │  │  ├─ 121_3.png
│  │     │  │  │  ├─ 121_4.png
│  │     │  │  │  ├─ 121_5.png
│  │     │  │  │  ├─ 121_6.png
│  │     │  │  │  ├─ 121_7.png
│  │     │  │  │  ├─ 122_0.png
│  │     │  │  │  ├─ 122_1.png
│  │     │  │  │  ├─ 122_2.png
│  │     │  │  │  ├─ 122_3.png
│  │     │  │  │  ├─ 122_4.png
│  │     │  │  │  ├─ 122_5.png
│  │     │  │  │  ├─ 122_6.png
│  │     │  │  │  ├─ 122_7.png
│  │     │  │  │  ├─ 123_0.png
│  │     │  │  │  ├─ 123_1.png
│  │     │  │  │  ├─ 123_2.png
│  │     │  │  │  ├─ 123_3.png
│  │     │  │  │  ├─ 123_4.png
│  │     │  │  │  ├─ 123_5.png
│  │     │  │  │  ├─ 123_6.png
│  │     │  │  │  ├─ 123_7.png
│  │     │  │  │  ├─ 124_0.png
│  │     │  │  │  ├─ 124_1.png
│  │     │  │  │  ├─ 124_2.png
│  │     │  │  │  ├─ 124_3.png
│  │     │  │  │  ├─ 124_4.png
│  │     │  │  │  ├─ 124_5.png
│  │     │  │  │  ├─ 124_6.png
│  │     │  │  │  ├─ 124_7.png
│  │     │  │  │  ├─ 125_0.png
│  │     │  │  │  ├─ 125_1.png
│  │     │  │  │  ├─ 125_2.png
│  │     │  │  │  ├─ 125_3.png
│  │     │  │  │  ├─ 125_4.png
│  │     │  │  │  ├─ 125_5.png
│  │     │  │  │  ├─ 125_6.png
│  │     │  │  │  ├─ 125_7.png
│  │     │  │  │  ├─ 126_0.png
│  │     │  │  │  ├─ 126_1.png
│  │     │  │  │  ├─ 126_2.png
│  │     │  │  │  ├─ 126_3.png
│  │     │  │  │  ├─ 126_4.png
│  │     │  │  │  ├─ 126_5.png
│  │     │  │  │  ├─ 126_6.png
│  │     │  │  │  ├─ 126_7.png
│  │     │  │  │  ├─ 32_0.png
│  │     │  │  │  ├─ 32_1.png
│  │     │  │  │  ├─ 32_2.png
│  │     │  │  │  ├─ 32_3.png
│  │     │  │  │  ├─ 32_4.png
│  │     │  │  │  ├─ 32_5.png
│  │     │  │  │  ├─ 32_6.png
│  │     │  │  │  ├─ 32_7.png
│  │     │  │  │  ├─ 33_0.png
│  │     │  │  │  ├─ 33_1.png
│  │     │  │  │  ├─ 33_2.png
│  │     │  │  │  ├─ 33_3.png
│  │     │  │  │  ├─ 33_4.png
│  │     │  │  │  ├─ 33_5.png
│  │     │  │  │  ├─ 33_6.png
│  │     │  │  │  ├─ 33_7.png
│  │     │  │  │  ├─ 34_0.png
│  │     │  │  │  ├─ 34_1.png
│  │     │  │  │  ├─ 34_2.png
│  │     │  │  │  ├─ 34_3.png
│  │     │  │  │  ├─ 34_4.png
│  │     │  │  │  ├─ 34_5.png
│  │     │  │  │  ├─ 34_6.png
│  │     │  │  │  ├─ 34_7.png
│  │     │  │  │  ├─ 35_0.png
│  │     │  │  │  ├─ 35_1.png
│  │     │  │  │  ├─ 35_2.png
│  │     │  │  │  ├─ 35_3.png
│  │     │  │  │  ├─ 35_4.png
│  │     │  │  │  ├─ 35_5.png
│  │     │  │  │  ├─ 35_6.png
│  │     │  │  │  ├─ 35_7.png
│  │     │  │  │  ├─ 36_0.png
│  │     │  │  │  ├─ 36_1.png
│  │     │  │  │  ├─ 36_2.png
│  │     │  │  │  ├─ 36_3.png
│  │     │  │  │  ├─ 36_4.png
│  │     │  │  │  ├─ 36_5.png
│  │     │  │  │  ├─ 36_6.png
│  │     │  │  │  ├─ 36_7.png
│  │     │  │  │  ├─ 37_0.png
│  │     │  │  │  ├─ 37_1.png
│  │     │  │  │  ├─ 37_2.png
│  │     │  │  │  ├─ 37_3.png
│  │     │  │  │  ├─ 37_4.png
│  │     │  │  │  ├─ 37_5.png
│  │     │  │  │  ├─ 37_6.png
│  │     │  │  │  ├─ 37_7.png
│  │     │  │  │  ├─ 38_0.png
│  │     │  │  │  ├─ 38_1.png
│  │     │  │  │  ├─ 38_2.png
│  │     │  │  │  ├─ 38_3.png
│  │     │  │  │  ├─ 38_4.png
│  │     │  │  │  ├─ 38_5.png
│  │     │  │  │  ├─ 38_6.png
│  │     │  │  │  ├─ 38_7.png
│  │     │  │  │  ├─ 39_0.png
│  │     │  │  │  ├─ 39_1.png
│  │     │  │  │  ├─ 39_2.png
│  │     │  │  │  ├─ 39_3.png
│  │     │  │  │  ├─ 39_4.png
│  │     │  │  │  ├─ 39_5.png
│  │     │  │  │  ├─ 39_6.png
│  │     │  │  │  ├─ 39_7.png
│  │     │  │  │  ├─ 40_0.png
│  │     │  │  │  ├─ 40_1.png
│  │     │  │  │  ├─ 40_2.png
│  │     │  │  │  ├─ 40_3.png
│  │     │  │  │  ├─ 40_4.png
│  │     │  │  │  ├─ 40_5.png
│  │     │  │  │  ├─ 40_6.png
│  │     │  │  │  ├─ 40_7.png
│  │     │  │  │  ├─ 41_0.png
│  │     │  │  │  ├─ 41_1.png
│  │     │  │  │  ├─ 41_2.png
│  │     │  │  │  ├─ 41_3.png
│  │     │  │  │  ├─ 41_4.png
│  │     │  │  │  ├─ 41_5.png
│  │     │  │  │  ├─ 41_6.png
│  │     │  │  │  ├─ 41_7.png
│  │     │  │  │  ├─ 42_0.png
│  │     │  │  │  ├─ 42_1.png
│  │     │  │  │  ├─ 42_2.png
│  │     │  │  │  ├─ 42_3.png
│  │     │  │  │  ├─ 42_4.png
│  │     │  │  │  ├─ 42_5.png
│  │     │  │  │  ├─ 42_6.png
│  │     │  │  │  ├─ 42_7.png
│  │     │  │  │  ├─ 43_0.png
│  │     │  │  │  ├─ 43_1.png
│  │     │  │  │  ├─ 43_2.png
│  │     │  │  │  ├─ 43_3.png
│  │     │  │  │  ├─ 43_4.png
│  │     │  │  │  ├─ 43_5.png
│  │     │  │  │  ├─ 43_6.png
│  │     │  │  │  ├─ 43_7.png
│  │     │  │  │  ├─ 44_0.png
│  │     │  │  │  ├─ 44_1.png
│  │     │  │  │  ├─ 44_2.png
│  │     │  │  │  ├─ 44_3.png
│  │     │  │  │  ├─ 44_4.png
│  │     │  │  │  ├─ 44_5.png
│  │     │  │  │  ├─ 44_6.png
│  │     │  │  │  ├─ 44_7.png
│  │     │  │  │  ├─ 45_0.png
│  │     │  │  │  ├─ 45_1.png
│  │     │  │  │  ├─ 45_2.png
│  │     │  │  │  ├─ 45_3.png
│  │     │  │  │  ├─ 45_4.png
│  │     │  │  │  ├─ 45_5.png
│  │     │  │  │  ├─ 45_6.png
│  │     │  │  │  ├─ 45_7.png
│  │     │  │  │  ├─ 46_0.png
│  │     │  │  │  ├─ 46_1.png
│  │     │  │  │  ├─ 46_2.png
│  │     │  │  │  ├─ 46_3.png
│  │     │  │  │  ├─ 46_4.png
│  │     │  │  │  ├─ 46_5.png
│  │     │  │  │  ├─ 46_6.png
│  │     │  │  │  ├─ 46_7.png
│  │     │  │  │  ├─ 47_0.png
│  │     │  │  │  ├─ 47_1.png
│  │     │  │  │  ├─ 47_2.png
│  │     │  │  │  ├─ 47_3.png
│  │     │  │  │  ├─ 47_4.png
│  │     │  │  │  ├─ 47_5.png
│  │     │  │  │  ├─ 47_6.png
│  │     │  │  │  ├─ 47_7.png
│  │     │  │  │  ├─ 48_0.png
│  │     │  │  │  ├─ 48_1.png
│  │     │  │  │  ├─ 48_2.png
│  │     │  │  │  ├─ 48_3.png
│  │     │  │  │  ├─ 48_4.png
│  │     │  │  │  ├─ 48_5.png
│  │     │  │  │  ├─ 48_6.png
│  │     │  │  │  ├─ 48_7.png
│  │     │  │  │  ├─ 49_0.png
│  │     │  │  │  ├─ 49_1.png
│  │     │  │  │  ├─ 49_2.png
│  │     │  │  │  ├─ 49_3.png
│  │     │  │  │  ├─ 49_4.png
│  │     │  │  │  ├─ 49_5.png
│  │     │  │  │  ├─ 49_6.png
│  │     │  │  │  ├─ 49_7.png
│  │     │  │  │  ├─ 50_0.png
│  │     │  │  │  ├─ 50_1.png
│  │     │  │  │  ├─ 50_2.png
│  │     │  │  │  ├─ 50_3.png
│  │     │  │  │  ├─ 50_4.png
│  │     │  │  │  ├─ 50_5.png
│  │     │  │  │  ├─ 50_6.png
│  │     │  │  │  ├─ 50_7.png
│  │     │  │  │  ├─ 51_0.png
│  │     │  │  │  ├─ 51_1.png
│  │     │  │  │  ├─ 51_2.png
│  │     │  │  │  ├─ 51_3.png
│  │     │  │  │  ├─ 51_4.png
│  │     │  │  │  ├─ 51_5.png
│  │     │  │  │  ├─ 51_6.png
│  │     │  │  │  ├─ 51_7.png
│  │     │  │  │  ├─ 52_0.png
│  │     │  │  │  ├─ 52_1.png
│  │     │  │  │  ├─ 52_2.png
│  │     │  │  │  ├─ 52_3.png
│  │     │  │  │  ├─ 52_4.png
│  │     │  │  │  ├─ 52_5.png
│  │     │  │  │  ├─ 52_6.png
│  │     │  │  │  ├─ 52_7.png
│  │     │  │  │  ├─ 53_0.png
│  │     │  │  │  ├─ 53_1.png
│  │     │  │  │  ├─ 53_2.png
│  │     │  │  │  ├─ 53_3.png
│  │     │  │  │  ├─ 53_4.png
│  │     │  │  │  ├─ 53_5.png
│  │     │  │  │  ├─ 53_6.png
│  │     │  │  │  ├─ 53_7.png
│  │     │  │  │  ├─ 54_0.png
│  │     │  │  │  ├─ 54_1.png
│  │     │  │  │  ├─ 54_2.png
│  │     │  │  │  ├─ 54_3.png
│  │     │  │  │  ├─ 54_4.png
│  │     │  │  │  ├─ 54_5.png
│  │     │  │  │  ├─ 54_6.png
│  │     │  │  │  ├─ 54_7.png
│  │     │  │  │  ├─ 55_0.png
│  │     │  │  │  ├─ 55_1.png
│  │     │  │  │  ├─ 55_2.png
│  │     │  │  │  ├─ 55_3.png
│  │     │  │  │  ├─ 55_4.png
│  │     │  │  │  ├─ 55_5.png
│  │     │  │  │  ├─ 55_6.png
│  │     │  │  │  ├─ 55_7.png
│  │     │  │  │  ├─ 56_0.png
│  │     │  │  │  ├─ 56_1.png
│  │     │  │  │  ├─ 56_2.png
│  │     │  │  │  ├─ 56_3.png
│  │     │  │  │  ├─ 56_4.png
│  │     │  │  │  ├─ 56_5.png
│  │     │  │  │  ├─ 56_6.png
│  │     │  │  │  ├─ 56_7.png
│  │     │  │  │  ├─ 57_0.png
│  │     │  │  │  ├─ 57_1.png
│  │     │  │  │  ├─ 57_2.png
│  │     │  │  │  ├─ 57_3.png
│  │     │  │  │  ├─ 57_4.png
│  │     │  │  │  ├─ 57_5.png
│  │     │  │  │  ├─ 57_6.png
│  │     │  │  │  ├─ 57_7.png
│  │     │  │  │  ├─ 58_0.png
│  │     │  │  │  ├─ 58_1.png
│  │     │  │  │  ├─ 58_2.png
│  │     │  │  │  ├─ 58_3.png
│  │     │  │  │  ├─ 58_4.png
│  │     │  │  │  ├─ 58_5.png
│  │     │  │  │  ├─ 58_6.png
│  │     │  │  │  ├─ 58_7.png
│  │     │  │  │  ├─ 59_0.png
│  │     │  │  │  ├─ 59_1.png
│  │     │  │  │  ├─ 59_2.png
│  │     │  │  │  ├─ 59_3.png
│  │     │  │  │  ├─ 59_4.png
│  │     │  │  │  ├─ 59_5.png
│  │     │  │  │  ├─ 59_6.png
│  │     │  │  │  ├─ 59_7.png
│  │     │  │  │  ├─ 60_0.png
│  │     │  │  │  ├─ 60_1.png
│  │     │  │  │  ├─ 60_2.png
│  │     │  │  │  ├─ 60_3.png
│  │     │  │  │  ├─ 60_4.png
│  │     │  │  │  ├─ 60_5.png
│  │     │  │  │  ├─ 60_6.png
│  │     │  │  │  ├─ 60_7.png
│  │     │  │  │  ├─ 61_0.png
│  │     │  │  │  ├─ 61_1.png
│  │     │  │  │  ├─ 61_2.png
│  │     │  │  │  ├─ 61_3.png
│  │     │  │  │  ├─ 61_4.png
│  │     │  │  │  ├─ 61_5.png
│  │     │  │  │  ├─ 61_6.png
│  │     │  │  │  ├─ 61_7.png
│  │     │  │  │  ├─ 62_0.png
│  │     │  │  │  ├─ 62_1.png
│  │     │  │  │  ├─ 62_2.png
│  │     │  │  │  ├─ 62_3.png
│  │     │  │  │  ├─ 62_4.png
│  │     │  │  │  ├─ 62_5.png
│  │     │  │  │  ├─ 62_6.png
│  │     │  │  │  ├─ 62_7.png
│  │     │  │  │  ├─ 63_0.png
│  │     │  │  │  ├─ 63_1.png
│  │     │  │  │  ├─ 63_2.png
│  │     │  │  │  ├─ 63_3.png
│  │     │  │  │  ├─ 63_4.png
│  │     │  │  │  ├─ 63_5.png
│  │     │  │  │  ├─ 63_6.png
│  │     │  │  │  ├─ 63_7.png
│  │     │  │  │  ├─ 64_0.png
│  │     │  │  │  ├─ 64_1.png
│  │     │  │  │  ├─ 64_2.png
│  │     │  │  │  ├─ 64_3.png
│  │     │  │  │  ├─ 64_4.png
│  │     │  │  │  ├─ 64_5.png
│  │     │  │  │  ├─ 64_6.png
│  │     │  │  │  ├─ 64_7.png
│  │     │  │  │  ├─ 65_0.png
│  │     │  │  │  ├─ 65_1.png
│  │     │  │  │  ├─ 65_2.png
│  │     │  │  │  ├─ 65_3.png
│  │     │  │  │  ├─ 65_4.png
│  │     │  │  │  ├─ 65_5.png
│  │     │  │  │  ├─ 65_6.png
│  │     │  │  │  ├─ 65_7.png
│  │     │  │  │  ├─ 66_0.png
│  │     │  │  │  ├─ 66_1.png
│  │     │  │  │  ├─ 66_2.png
│  │     │  │  │  ├─ 66_3.png
│  │     │  │  │  ├─ 66_4.png
│  │     │  │  │  ├─ 66_5.png
│  │     │  │  │  ├─ 66_6.png
│  │     │  │  │  ├─ 66_7.png
│  │     │  │  │  ├─ 67_0.png
│  │     │  │  │  ├─ 67_1.png
│  │     │  │  │  ├─ 67_2.png
│  │     │  │  │  ├─ 67_3.png
│  │     │  │  │  ├─ 67_4.png
│  │     │  │  │  ├─ 67_5.png
│  │     │  │  │  ├─ 67_6.png
│  │     │  │  │  ├─ 67_7.png
│  │     │  │  │  ├─ 68_0.png
│  │     │  │  │  ├─ 68_1.png
│  │     │  │  │  ├─ 68_2.png
│  │     │  │  │  ├─ 68_3.png
│  │     │  │  │  ├─ 68_4.png
│  │     │  │  │  ├─ 68_5.png
│  │     │  │  │  ├─ 68_6.png
│  │     │  │  │  ├─ 68_7.png
│  │     │  │  │  ├─ 69_0.png
│  │     │  │  │  ├─ 69_1.png
│  │     │  │  │  ├─ 69_2.png
│  │     │  │  │  ├─ 69_3.png
│  │     │  │  │  ├─ 69_4.png
│  │     │  │  │  ├─ 69_5.png
│  │     │  │  │  ├─ 69_6.png
│  │     │  │  │  ├─ 69_7.png
│  │     │  │  │  ├─ 70_0.png
│  │     │  │  │  ├─ 70_1.png
│  │     │  │  │  ├─ 70_2.png
│  │     │  │  │  ├─ 70_3.png
│  │     │  │  │  ├─ 70_4.png
│  │     │  │  │  ├─ 70_5.png
│  │     │  │  │  ├─ 70_6.png
│  │     │  │  │  ├─ 70_7.png
│  │     │  │  │  ├─ 71_0.png
│  │     │  │  │  ├─ 71_1.png
│  │     │  │  │  ├─ 71_2.png
│  │     │  │  │  ├─ 71_3.png
│  │     │  │  │  ├─ 71_4.png
│  │     │  │  │  ├─ 71_5.png
│  │     │  │  │  ├─ 71_6.png
│  │     │  │  │  ├─ 71_7.png
│  │     │  │  │  ├─ 72_0.png
│  │     │  │  │  ├─ 72_1.png
│  │     │  │  │  ├─ 72_2.png
│  │     │  │  │  ├─ 72_3.png
│  │     │  │  │  ├─ 72_4.png
│  │     │  │  │  ├─ 72_5.png
│  │     │  │  │  ├─ 72_6.png
│  │     │  │  │  ├─ 72_7.png
│  │     │  │  │  ├─ 73_0.png
│  │     │  │  │  ├─ 73_1.png
│  │     │  │  │  ├─ 73_2.png
│  │     │  │  │  ├─ 73_3.png
│  │     │  │  │  ├─ 73_4.png
│  │     │  │  │  ├─ 73_5.png
│  │     │  │  │  ├─ 73_6.png
│  │     │  │  │  ├─ 73_7.png
│  │     │  │  │  ├─ 74_0.png
│  │     │  │  │  ├─ 74_1.png
│  │     │  │  │  ├─ 74_2.png
│  │     │  │  │  ├─ 74_3.png
│  │     │  │  │  ├─ 74_4.png
│  │     │  │  │  ├─ 74_5.png
│  │     │  │  │  ├─ 74_6.png
│  │     │  │  │  ├─ 74_7.png
│  │     │  │  │  ├─ 75_0.png
│  │     │  │  │  ├─ 75_1.png
│  │     │  │  │  ├─ 75_2.png
│  │     │  │  │  ├─ 75_3.png
│  │     │  │  │  ├─ 75_4.png
│  │     │  │  │  ├─ 75_5.png
│  │     │  │  │  ├─ 75_6.png
│  │     │  │  │  ├─ 75_7.png
│  │     │  │  │  ├─ 76_0.png
│  │     │  │  │  ├─ 76_1.png
│  │     │  │  │  ├─ 76_2.png
│  │     │  │  │  ├─ 76_3.png
│  │     │  │  │  ├─ 76_4.png
│  │     │  │  │  ├─ 76_5.png
│  │     │  │  │  ├─ 76_6.png
│  │     │  │  │  ├─ 76_7.png
│  │     │  │  │  ├─ 77_0.png
│  │     │  │  │  ├─ 77_1.png
│  │     │  │  │  ├─ 77_2.png
│  │     │  │  │  ├─ 77_3.png
│  │     │  │  │  ├─ 77_4.png
│  │     │  │  │  ├─ 77_5.png
│  │     │  │  │  ├─ 77_6.png
│  │     │  │  │  ├─ 77_7.png
│  │     │  │  │  ├─ 78_0.png
│  │     │  │  │  ├─ 78_1.png
│  │     │  │  │  ├─ 78_2.png
│  │     │  │  │  ├─ 78_3.png
│  │     │  │  │  ├─ 78_4.png
│  │     │  │  │  ├─ 78_5.png
│  │     │  │  │  ├─ 78_6.png
│  │     │  │  │  ├─ 78_7.png
│  │     │  │  │  ├─ 79_0.png
│  │     │  │  │  ├─ 79_1.png
│  │     │  │  │  ├─ 79_2.png
│  │     │  │  │  ├─ 79_3.png
│  │     │  │  │  ├─ 79_4.png
│  │     │  │  │  ├─ 79_5.png
│  │     │  │  │  ├─ 79_6.png
│  │     │  │  │  ├─ 79_7.png
│  │     │  │  │  ├─ 80_0.png
│  │     │  │  │  ├─ 80_1.png
│  │     │  │  │  ├─ 80_2.png
│  │     │  │  │  ├─ 80_3.png
│  │     │  │  │  ├─ 80_4.png
│  │     │  │  │  ├─ 80_5.png
│  │     │  │  │  ├─ 80_6.png
│  │     │  │  │  ├─ 80_7.png
│  │     │  │  │  ├─ 81_0.png
│  │     │  │  │  ├─ 81_1.png
│  │     │  │  │  ├─ 81_2.png
│  │     │  │  │  ├─ 81_3.png
│  │     │  │  │  ├─ 81_4.png
│  │     │  │  │  ├─ 81_5.png
│  │     │  │  │  ├─ 81_6.png
│  │     │  │  │  ├─ 81_7.png
│  │     │  │  │  ├─ 82_0.png
│  │     │  │  │  ├─ 82_1.png
│  │     │  │  │  ├─ 82_2.png
│  │     │  │  │  ├─ 82_3.png
│  │     │  │  │  ├─ 82_4.png
│  │     │  │  │  ├─ 82_5.png
│  │     │  │  │  ├─ 82_6.png
│  │     │  │  │  ├─ 82_7.png
│  │     │  │  │  ├─ 83_0.png
│  │     │  │  │  ├─ 83_1.png
│  │     │  │  │  ├─ 83_2.png
│  │     │  │  │  ├─ 83_3.png
│  │     │  │  │  ├─ 83_4.png
│  │     │  │  │  ├─ 83_5.png
│  │     │  │  │  ├─ 83_6.png
│  │     │  │  │  ├─ 83_7.png
│  │     │  │  │  ├─ 84_0.png
│  │     │  │  │  ├─ 84_1.png
│  │     │  │  │  ├─ 84_2.png
│  │     │  │  │  ├─ 84_3.png
│  │     │  │  │  ├─ 84_4.png
│  │     │  │  │  ├─ 84_5.png
│  │     │  │  │  ├─ 84_6.png
│  │     │  │  │  ├─ 84_7.png
│  │     │  │  │  ├─ 85_0.png
│  │     │  │  │  ├─ 85_1.png
│  │     │  │  │  ├─ 85_2.png
│  │     │  │  │  ├─ 85_3.png
│  │     │  │  │  ├─ 85_4.png
│  │     │  │  │  ├─ 85_5.png
│  │     │  │  │  ├─ 85_6.png
│  │     │  │  │  ├─ 85_7.png
│  │     │  │  │  ├─ 86_0.png
│  │     │  │  │  ├─ 86_1.png
│  │     │  │  │  ├─ 86_2.png
│  │     │  │  │  ├─ 86_3.png
│  │     │  │  │  ├─ 86_4.png
│  │     │  │  │  ├─ 86_5.png
│  │     │  │  │  ├─ 86_6.png
│  │     │  │  │  ├─ 86_7.png
│  │     │  │  │  ├─ 87_0.png
│  │     │  │  │  ├─ 87_1.png
│  │     │  │  │  ├─ 87_2.png
│  │     │  │  │  ├─ 87_3.png
│  │     │  │  │  ├─ 87_4.png
│  │     │  │  │  ├─ 87_5.png
│  │     │  │  │  ├─ 87_6.png
│  │     │  │  │  ├─ 87_7.png
│  │     │  │  │  ├─ 88_0.png
│  │     │  │  │  ├─ 88_1.png
│  │     │  │  │  ├─ 88_2.png
│  │     │  │  │  ├─ 88_3.png
│  │     │  │  │  ├─ 88_4.png
│  │     │  │  │  ├─ 88_5.png
│  │     │  │  │  ├─ 88_6.png
│  │     │  │  │  ├─ 88_7.png
│  │     │  │  │  ├─ 89_0.png
│  │     │  │  │  ├─ 89_1.png
│  │     │  │  │  ├─ 89_2.png
│  │     │  │  │  ├─ 89_3.png
│  │     │  │  │  ├─ 89_4.png
│  │     │  │  │  ├─ 89_5.png
│  │     │  │  │  ├─ 89_6.png
│  │     │  │  │  ├─ 89_7.png
│  │     │  │  │  ├─ 90_0.png
│  │     │  │  │  ├─ 90_1.png
│  │     │  │  │  ├─ 90_2.png
│  │     │  │  │  ├─ 90_3.png
│  │     │  │  │  ├─ 90_4.png
│  │     │  │  │  ├─ 90_5.png
│  │     │  │  │  ├─ 90_6.png
│  │     │  │  │  ├─ 90_7.png
│  │     │  │  │  ├─ 91_0.png
│  │     │  │  │  ├─ 91_1.png
│  │     │  │  │  ├─ 91_2.png
│  │     │  │  │  ├─ 91_3.png
│  │     │  │  │  ├─ 91_4.png
│  │     │  │  │  ├─ 91_5.png
│  │     │  │  │  ├─ 91_6.png
│  │     │  │  │  ├─ 91_7.png
│  │     │  │  │  ├─ 92_0.png
│  │     │  │  │  ├─ 92_1.png
│  │     │  │  │  ├─ 92_2.png
│  │     │  │  │  ├─ 92_3.png
│  │     │  │  │  ├─ 92_4.png
│  │     │  │  │  ├─ 92_5.png
│  │     │  │  │  ├─ 92_6.png
│  │     │  │  │  ├─ 92_7.png
│  │     │  │  │  ├─ 93_0.png
│  │     │  │  │  ├─ 93_1.png
│  │     │  │  │  ├─ 93_2.png
│  │     │  │  │  ├─ 93_3.png
│  │     │  │  │  ├─ 93_4.png
│  │     │  │  │  ├─ 93_5.png
│  │     │  │  │  ├─ 93_6.png
│  │     │  │  │  ├─ 93_7.png
│  │     │  │  │  ├─ 94_0.png
│  │     │  │  │  ├─ 94_1.png
│  │     │  │  │  ├─ 94_2.png
│  │     │  │  │  ├─ 94_3.png
│  │     │  │  │  ├─ 94_4.png
│  │     │  │  │  ├─ 94_5.png
│  │     │  │  │  ├─ 94_6.png
│  │     │  │  │  ├─ 94_7.png
│  │     │  │  │  ├─ 95_0.png
│  │     │  │  │  ├─ 95_1.png
│  │     │  │  │  ├─ 95_2.png
│  │     │  │  │  ├─ 95_3.png
│  │     │  │  │  ├─ 95_4.png
│  │     │  │  │  ├─ 95_5.png
│  │     │  │  │  ├─ 95_6.png
│  │     │  │  │  ├─ 95_7.png
│  │     │  │  │  ├─ 96_0.png
│  │     │  │  │  ├─ 96_1.png
│  │     │  │  │  ├─ 96_2.png
│  │     │  │  │  ├─ 96_3.png
│  │     │  │  │  ├─ 96_4.png
│  │     │  │  │  ├─ 96_5.png
│  │     │  │  │  ├─ 96_6.png
│  │     │  │  │  ├─ 96_7.png
│  │     │  │  │  ├─ 97_0.png
│  │     │  │  │  ├─ 97_1.png
│  │     │  │  │  ├─ 97_2.png
│  │     │  │  │  ├─ 97_3.png
│  │     │  │  │  ├─ 97_4.png
│  │     │  │  │  ├─ 97_5.png
│  │     │  │  │  ├─ 97_6.png
│  │     │  │  │  ├─ 97_7.png
│  │     │  │  │  ├─ 98_0.png
│  │     │  │  │  ├─ 98_1.png
│  │     │  │  │  ├─ 98_2.png
│  │     │  │  │  ├─ 98_3.png
│  │     │  │  │  ├─ 98_4.png
│  │     │  │  │  ├─ 98_5.png
│  │     │  │  │  ├─ 98_6.png
│  │     │  │  │  ├─ 98_7.png
│  │     │  │  │  ├─ 99_0.png
│  │     │  │  │  ├─ 99_1.png
│  │     │  │  │  ├─ 99_2.png
│  │     │  │  │  ├─ 99_3.png
│  │     │  │  │  ├─ 99_4.png
│  │     │  │  │  ├─ 99_5.png
│  │     │  │  │  ├─ 99_6.png
│  │     │  │  │  ├─ 99_7.png
│  │     │  │  │  └─ make_labels.py
│  │     │  │  ├─ openimages.data
│  │     │  │  ├─ openimages.names
│  │     │  │  ├─ person.jpg
│  │     │  │  ├─ scream.jpg
│  │     │  │  ├─ voc
│  │     │  │  │  └─ voc_label.py
│  │     │  │  ├─ voc.data
│  │     │  │  └─ voc.names
│  │     │  ├─ densenet201_yolo.cfg
│  │     │  ├─ dog.jpg
│  │     │  ├─ dogr.jpg
│  │     │  ├─ gen_anchors.py
│  │     │  ├─ partial.cmd
│  │     │  ├─ pthreadGC2.dll
│  │     │  ├─ pthreadVC2.dll
│  │     │  ├─ resnet152_yolo.cfg
│  │     │  ├─ resnet50_yolo.cfg
│  │     │  ├─ reval_voc_py3.py
│  │     │  ├─ rnn_lstm.cmd
│  │     │  ├─ rnn_tolstoy.cmd
│  │     │  ├─ tiny-yolo-voc.cfg
│  │     │  ├─ tiny-yolo.cfg
│  │     │  ├─ train_voc.cmd
│  │     │  ├─ voc_eval_py3.py
│  │     │  ├─ yolo-voc.2.0.cfg
│  │     │  ├─ yolo-voc.cfg
│  │     │  ├─ yolo.2.0.cfg
│  │     │  ├─ yolo.cfg
│  │     │  ├─ yolo9000.cfg
│  │     │  ├─ yolov3-voc.cfg
│  │     │  └─ yolov3.cfg
│  │     ├─ yolo_console_dll.sln
│  │     ├─ yolo_console_dll.vcxproj
│  │     ├─ yolo_cpp_dll.sln
│  │     ├─ yolo_cpp_dll.vcxproj
│  │     ├─ yolo_cpp_dll_no_gpu.sln
│  │     └─ yolo_cpp_dll_no_gpu.vcxproj
│  ├─ build.ps1
│  ├─ build.sh
│  ├─ cfg
│  │  ├─ 9k.labels
│  │  ├─ 9k.names
│  │  ├─ 9k.tree
│  │  ├─ Gaussian_yolov3_BDD.cfg
│  │  ├─ alexnet.cfg
│  │  ├─ cd53paspp-gamma.cfg
│  │  ├─ cifar.cfg
│  │  ├─ cifar.test.cfg
│  │  ├─ coco.data
│  │  ├─ coco.names
│  │  ├─ coco9k.map
│  │  ├─ combine9k.data
│  │  ├─ crnn.train.cfg
│  │  ├─ csdarknet53-omega.cfg
│  │  ├─ cspx-p7-mish-omega.cfg
│  │  ├─ cspx-p7-mish_hp.cfg
│  │  ├─ csresnext50-panet-spp-original-optimal.cfg
│  │  ├─ csresnext50-panet-spp.cfg
│  │  ├─ darknet.cfg
│  │  ├─ darknet19.cfg
│  │  ├─ darknet19_448.cfg
│  │  ├─ darknet53.cfg
│  │  ├─ darknet53_448_xnor.cfg
│  │  ├─ densenet201.cfg
│  │  ├─ efficientnet-lite3.cfg
│  │  ├─ efficientnet_b0.cfg
│  │  ├─ enet-coco.cfg
│  │  ├─ extraction.cfg
│  │  ├─ extraction.conv.cfg
│  │  ├─ extraction22k.cfg
│  │  ├─ go.test.cfg
│  │  ├─ gru.cfg
│  │  ├─ imagenet.labels.list
│  │  ├─ imagenet.shortnames.list
│  │  ├─ imagenet1k.data
│  │  ├─ imagenet22k.dataset
│  │  ├─ imagenet9k.hierarchy.dataset
│  │  ├─ inet9k.map
│  │  ├─ jnet-conv.cfg
│  │  ├─ lstm.train.cfg
│  │  ├─ openimages.data
│  │  ├─ resnet101.cfg
│  │  ├─ resnet152.cfg
│  │  ├─ resnet152_trident.cfg
│  │  ├─ resnet50.cfg
│  │  ├─ resnext152-32x4d.cfg
│  │  ├─ rnn.cfg
│  │  ├─ rnn.train.cfg
│  │  ├─ strided.cfg
│  │  ├─ t1.test.cfg
│  │  ├─ tiny-yolo-voc.cfg
│  │  ├─ tiny-yolo.cfg
│  │  ├─ tiny-yolo_xnor.cfg
│  │  ├─ tiny.cfg
│  │  ├─ vgg-16.cfg
│  │  ├─ vgg-conv.cfg
│  │  ├─ voc.data
│  │  ├─ writing.cfg
│  │  ├─ yolo-voc.2.0.cfg
│  │  ├─ yolo-voc.cfg
│  │  ├─ yolo.2.0.cfg
│  │  ├─ yolo.cfg
│  │  ├─ yolo9000.cfg
│  │  ├─ yolov1
│  │  │  ├─ tiny-coco.cfg
│  │  │  ├─ tiny-yolo.cfg
│  │  │  ├─ xyolo.test.cfg
│  │  │  ├─ yolo-coco.cfg
│  │  │  ├─ yolo-small.cfg
│  │  │  ├─ yolo.cfg
│  │  │  ├─ yolo.train.cfg
│  │  │  └─ yolo2.cfg
│  │  ├─ yolov2-tiny-voc.cfg
│  │  ├─ yolov2-tiny.cfg
│  │  ├─ yolov2-voc.cfg
│  │  ├─ yolov2.cfg
│  │  ├─ yolov3-openimages.cfg
│  │  ├─ yolov3-spp.cfg
│  │  ├─ yolov3-tiny-prn.cfg
│  │  ├─ yolov3-tiny.cfg
│  │  ├─ yolov3-tiny_3l.cfg
│  │  ├─ yolov3-tiny_obj.cfg
│  │  ├─ yolov3-tiny_occlusion_track.cfg
│  │  ├─ yolov3-tiny_xnor.cfg
│  │  ├─ yolov3-voc.cfg
│  │  ├─ yolov3-voc.yolov3-giou-40.cfg
│  │  ├─ yolov3.cfg
│  │  ├─ yolov3.coco-giou-12.cfg
│  │  ├─ yolov3_5l.cfg
│  │  ├─ yolov4-custom.cfg
│  │  ├─ yolov4-tiny-3l.cfg
│  │  ├─ yolov4-tiny-custom.cfg
│  │  ├─ yolov4-tiny.cfg
│  │  ├─ yolov4-tiny_contrastive.cfg
│  │  └─ yolov4.cfg
│  ├─ cmake
│  │  ├─ Modules
│  │  │  ├─ FindCUDNN.cmake
│  │  │  ├─ FindPThreads_windows.cmake
│  │  │  └─ FindStb.cmake
│  │  ├─ vcpkg_linux.diff
│  │  ├─ vcpkg_linux_cuda.diff
│  │  ├─ vcpkg_osx.diff
│  │  ├─ vcpkg_windows.diff
│  │  └─ vcpkg_windows_cuda.diff
│  ├─ darknet
│  ├─ darknet.py
│  ├─ darknet_images.py
│  ├─ darknet_video.py
│  ├─ image_yolov3.sh
│  ├─ image_yolov4.sh
│  ├─ include
│  │  ├─ darknet.h
│  │  └─ yolo_v2_class.hpp
│  ├─ json_mjpeg_streams.sh
│  ├─ libdarknet.so
│  ├─ net_cam_v3.sh
│  ├─ net_cam_v4.sh
│  ├─ obj
│  │  ├─ activation_kernels.o
│  │  ├─ activation_layer.o
│  │  ├─ activations.o
│  │  ├─ art.o
│  │  ├─ avgpool_layer.o
│  │  ├─ avgpool_layer_kernels.o
│  │  ├─ batchnorm_layer.o
│  │  ├─ blas.o
│  │  ├─ blas_kernels.o
│  │  ├─ box.o
│  │  ├─ captcha.o
│  │  ├─ cifar.o
│  │  ├─ classifier.o
│  │  ├─ coco.o
│  │  ├─ col2im.o
│  │  ├─ col2im_kernels.o
│  │  ├─ compare.o
│  │  ├─ connected_layer.o
│  │  ├─ conv_lstm_layer.o
│  │  ├─ convolutional_kernels.o
│  │  ├─ convolutional_layer.o
│  │  ├─ cost_layer.o
│  │  ├─ crnn_layer.o
│  │  ├─ crop_layer.o
│  │  ├─ crop_layer_kernels.o
│  │  ├─ dark_cuda.o
│  │  ├─ darknet.o
│  │  ├─ data.o
│  │  ├─ demo.o
│  │  ├─ detection_layer.o
│  │  ├─ detector.o
│  │  ├─ dice.o
│  │  ├─ dropout_layer.o
│  │  ├─ dropout_layer_kernels.o
│  │  ├─ gaussian_yolo_layer.o
│  │  ├─ gemm.o
│  │  ├─ go.o
│  │  ├─ gru_layer.o
│  │  ├─ http_stream.o
│  │  ├─ im2col.o
│  │  ├─ im2col_kernels.o
│  │  ├─ image.o
│  │  ├─ image_opencv.o
│  │  ├─ layer.o
│  │  ├─ list.o
│  │  ├─ local_layer.o
│  │  ├─ lstm_layer.o
│  │  ├─ matrix.o
│  │  ├─ maxpool_layer.o
│  │  ├─ maxpool_layer_kernels.o
│  │  ├─ network.o
│  │  ├─ network_kernels.o
│  │  ├─ nightmare.o
│  │  ├─ normalization_layer.o
│  │  ├─ option_list.o
│  │  ├─ parser.o
│  │  ├─ region_layer.o
│  │  ├─ reorg_layer.o
│  │  ├─ reorg_old_layer.o
│  │  ├─ rnn.o
│  │  ├─ rnn_layer.o
│  │  ├─ rnn_vid.o
│  │  ├─ route_layer.o
│  │  ├─ sam_layer.o
│  │  ├─ scale_channels_layer.o
│  │  ├─ shortcut_layer.o
│  │  ├─ softmax_layer.o
│  │  ├─ super.o
│  │  ├─ swag.o
│  │  ├─ tag.o
│  │  ├─ tree.o
│  │  ├─ upsample_layer.o
│  │  ├─ utils.o
│  │  ├─ voxel.o
│  │  ├─ writing.o
│  │  ├─ yolo.o
│  │  └─ yolo_layer.o
│  ├─ results
│  │  └─ comp4_det_test_aircraft.txt
│  ├─ scripts
│  │  ├─ README.md
│  │  ├─ dice_label.sh
│  │  ├─ gen_anchors.py
│  │  ├─ gen_tactic.sh
│  │  ├─ get_coco2017.sh
│  │  ├─ get_coco_dataset.sh
│  │  ├─ get_imagenet_train.sh
│  │  ├─ get_openimages_dataset.py
│  │  ├─ imagenet_label.sh
│  │  ├─ install_OpenCV4.sh
│  │  ├─ kitti2yolo.py
│  │  ├─ kmeansiou.c
│  │  ├─ log_parser
│  │  │  ├─ log_parser.py
│  │  │  ├─ plot.jpg
│  │  │  ├─ readme.md
│  │  │  ├─ run_log_parser_windows.cmd
│  │  │  ├─ test.log
│  │  │  ├─ test_new.log
│  │  │  └─ test_new.svg
│  │  ├─ reval_voc.py
│  │  ├─ reval_voc_py3.py
│  │  ├─ setup.ps1
│  │  ├─ setup.sh
│  │  ├─ testdev2017.txt
│  │  ├─ voc_eval.py
│  │  ├─ voc_eval_py3.py
│  │  ├─ voc_label.py
│  │  ├─ voc_label_difficult.py
│  │  └─ windows
│  │     ├─ otb_get_labels.sh
│  │     ├─ win_cifar.cmd
│  │     ├─ win_get_imagenet_train_48hours.cmd
│  │     ├─ win_get_imagenet_valid.cmd
│  │     ├─ win_get_otb_datasets.cmd
│  │     ├─ win_install_cygwin.cmd
│  │     ├─ windows_imagenet_label.sh
│  │     └─ windows_imagenet_train.sh
│  ├─ src
│  │  ├─ activation_kernels.cu
│  │  ├─ activation_layer.c
│  │  ├─ activation_layer.h
│  │  ├─ activations.c
│  │  ├─ activations.h
│  │  ├─ art.c
│  │  ├─ avgpool_layer.c
│  │  ├─ avgpool_layer.h
│  │  ├─ avgpool_layer_kernels.cu
│  │  ├─ batchnorm_layer.c
│  │  ├─ batchnorm_layer.h
│  │  ├─ blas.c
│  │  ├─ blas.h
│  │  ├─ blas_kernels.cu
│  │  ├─ box.c
│  │  ├─ box.h
│  │  ├─ captcha.c
│  │  ├─ cifar.c
│  │  ├─ classifier.c
│  │  ├─ classifier.h
│  │  ├─ coco.c
│  │  ├─ col2im.c
│  │  ├─ col2im.h
│  │  ├─ col2im_kernels.cu
│  │  ├─ compare.c
│  │  ├─ connected_layer.c
│  │  ├─ connected_layer.h
│  │  ├─ conv_lstm_layer.c
│  │  ├─ conv_lstm_layer.h
│  │  ├─ convolutional_kernels.cu
│  │  ├─ convolutional_layer.c
│  │  ├─ convolutional_layer.h
│  │  ├─ cost_layer.c
│  │  ├─ cost_layer.h
│  │  ├─ cpu_gemm.c
│  │  ├─ crnn_layer.c
│  │  ├─ crnn_layer.h
│  │  ├─ crop_layer.c
│  │  ├─ crop_layer.h
│  │  ├─ crop_layer_kernels.cu
│  │  ├─ dark_cuda.c
│  │  ├─ dark_cuda.h
│  │  ├─ darknet.c
│  │  ├─ darkunistd.h
│  │  ├─ data.c
│  │  ├─ data.h
│  │  ├─ deconvolutional_kernels.cu
│  │  ├─ deconvolutional_layer.c
│  │  ├─ deconvolutional_layer.h
│  │  ├─ demo.c
│  │  ├─ demo.h
│  │  ├─ detection_layer.c
│  │  ├─ detection_layer.h
│  │  ├─ detector.c
│  │  ├─ dice.c
│  │  ├─ dropout_layer.c
│  │  ├─ dropout_layer.h
│  │  ├─ dropout_layer_kernels.cu
│  │  ├─ gaussian_yolo_layer.c
│  │  ├─ gaussian_yolo_layer.h
│  │  ├─ gemm.c
│  │  ├─ gemm.h
│  │  ├─ getopt.c
│  │  ├─ getopt.h
│  │  ├─ gettimeofday.c
│  │  ├─ gettimeofday.h
│  │  ├─ go.c
│  │  ├─ gru_layer.c
│  │  ├─ gru_layer.h
│  │  ├─ http_stream.cpp
│  │  ├─ http_stream.h
│  │  ├─ httplib.h
│  │  ├─ im2col.c
│  │  ├─ im2col.h
│  │  ├─ im2col_kernels.cu
│  │  ├─ image.c
│  │  ├─ image.h
│  │  ├─ image_opencv.cpp
│  │  ├─ image_opencv.h
│  │  ├─ layer.c
│  │  ├─ layer.h
│  │  ├─ list.c
│  │  ├─ list.h
│  │  ├─ local_layer.c
│  │  ├─ local_layer.h
│  │  ├─ lstm_layer.c
│  │  ├─ lstm_layer.h
│  │  ├─ matrix.c
│  │  ├─ matrix.h
│  │  ├─ maxpool_layer.c
│  │  ├─ maxpool_layer.h
│  │  ├─ maxpool_layer_kernels.cu
│  │  ├─ network.c
│  │  ├─ network.h
│  │  ├─ network_kernels.cu
│  │  ├─ nightmare.c
│  │  ├─ normalization_layer.c
│  │  ├─ normalization_layer.h
│  │  ├─ option_list.c
│  │  ├─ option_list.h
│  │  ├─ parser.c
│  │  ├─ parser.h
│  │  ├─ region_layer.c
│  │  ├─ region_layer.h
│  │  ├─ reorg_layer.c
│  │  ├─ reorg_layer.h
│  │  ├─ reorg_old_layer.c
│  │  ├─ reorg_old_layer.h
│  │  ├─ rnn.c
│  │  ├─ rnn_layer.c
│  │  ├─ rnn_layer.h
│  │  ├─ rnn_vid.c
│  │  ├─ route_layer.c
│  │  ├─ route_layer.h
│  │  ├─ sam_layer.c
│  │  ├─ sam_layer.h
│  │  ├─ scale_channels_layer.c
│  │  ├─ scale_channels_layer.h
│  │  ├─ shortcut_layer.c
│  │  ├─ shortcut_layer.h
│  │  ├─ softmax_layer.c
│  │  ├─ softmax_layer.h
│  │  ├─ super.c
│  │  ├─ swag.c
│  │  ├─ tag.c
│  │  ├─ tree.c
│  │  ├─ tree.h
│  │  ├─ upsample_layer.c
│  │  ├─ upsample_layer.h
│  │  ├─ utils.c
│  │  ├─ utils.h
│  │  ├─ version.h
│  │  ├─ version.h.in
│  │  ├─ voxel.c
│  │  ├─ writing.c
│  │  ├─ yolo.c
│  │  ├─ yolo_console_dll.cpp
│  │  ├─ yolo_layer.c
│  │  ├─ yolo_layer.h
│  │  └─ yolo_v2_class.cpp
│  ├─ uselib
│  ├─ video_yolov3.sh
│  └─ video_yolov4.sh
├─ merge_result.geojson
├─ merge_result.tif
└─ run.py

```
```
Rs-Obj-Detection-Api
├─ .vscode
│  └─ settings.json
├─ README.md
├─ __pycache__
│  └─ config.cpython-36.pyc
├─ app
│  ├─ __init__.py
│  ├─ __pycache__
│  │  ├─ __init__.cpython-36.pyc
│  │  ├─ models.cpython-36.pyc
│  │  └─ request.cpython-36.pyc
│  ├─ detection
│  │  ├─ __init__.py
│  │  ├─ __pycache__
│  │  │  ├─ __init__.cpython-36.pyc
│  │  │  ├─ config.cpython-36.pyc
│  │  │  ├─ darknet.cpython-36.pyc
│  │  │  ├─ errors.cpython-36.pyc
│  │  │  ├─ request.cpython-36.pyc
│  │  │  ├─ requests.cpython-36.pyc
│  │  │  ├─ utils.cpython-36.pyc
│  │  │  └─ views.cpython-36.pyc
│  │  ├─ core
│  │  │  ├─ __init__.py
│  │  │  ├─ __pycache__
│  │  │  │  ├─ __init__.cpython-36.pyc
│  │  │  │  ├─ add_georeference.cpython-36.pyc
│  │  │  │  ├─ non_max_suppression.cpython-36.pyc
│  │  │  │  ├─ post_process.cpython-36.pyc
│  │  │  │  └─ slice_ims.cpython-36.pyc
│  │  │  ├─ add_georeference.py
│  │  │  ├─ non_max_suppression.py
│  │  │  ├─ post_process.py
│  │  │  └─ slice_ims.py
│  │  ├─ darknet.py
│  │  ├─ errors.py
│  │  ├─ utils.py
│  │  └─ views.py
│  ├─ models.py
│  ├─ request.py
│  └─ segmentation
│     ├─ __init__.py
│     ├─ __pycache__
│     │  ├─ __init__.cpython-36.pyc
│     │  ├─ utils.cpython-36.pyc
│     │  └─ views.cpython-36.pyc
│     ├─ segmentation_models_pytorch
│     │  ├─ __init__.py
│     │  ├─ __pycache__
│     │  │  ├─ __init__.cpython-36.pyc
│     │  │  ├─ __init__.cpython-37.pyc
│     │  │  └─ __version__.cpython-36.pyc
│     │  ├─ __version__.py
│     │  ├─ base
│     │  │  ├─ __init__.py
│     │  │  ├─ __pycache__
│     │  │  │  ├─ __init__.cpython-36.pyc
│     │  │  │  ├─ heads.cpython-36.pyc
│     │  │  │  ├─ initialization.cpython-36.pyc
│     │  │  │  ├─ model.cpython-36.pyc
│     │  │  │  └─ modules.cpython-36.pyc
│     │  │  ├─ heads.py
│     │  │  ├─ initialization.py
│     │  │  ├─ model.py
│     │  │  └─ modules.py
│     │  ├─ encoders
│     │  │  ├─ _EfficientNet-PyTorch
│     │  │  │  ├─ LICENSE
│     │  │  │  ├─ README.md
│     │  │  │  ├─ examples
│     │  │  │  │  ├─ imagenet
│     │  │  │  │  │  ├─ README.md
│     │  │  │  │  │  ├─ data
│     │  │  │  │  │  │  └─ README.md
│     │  │  │  │  │  └─ main.py
│     │  │  │  │  └─ simple
│     │  │  │  │     ├─ check.ipynb
│     │  │  │  │     ├─ example.ipynb
│     │  │  │  │     ├─ img.jpg
│     │  │  │  │     ├─ img2.jpg
│     │  │  │  │     └─ labels_map.txt
│     │  │  │  ├─ hubconf.py
│     │  │  │  ├─ setup.py
│     │  │  │  ├─ tests
│     │  │  │  │  └─ test_model.py
│     │  │  │  └─ tf_to_pytorch
│     │  │  │     ├─ README.md
│     │  │  │     ├─ convert_tf_to_pt
│     │  │  │     │  ├─ download.sh
│     │  │  │     │  ├─ load_tf_weights.py
│     │  │  │     │  ├─ load_tf_weights_tf1.py
│     │  │  │     │  ├─ original_tf
│     │  │  │     │  │  ├─ __init__.py
│     │  │  │     │  │  ├─ efficientnet_builder.py
│     │  │  │     │  │  ├─ efficientnet_model.py
│     │  │  │     │  │  ├─ eval_ckpt_main.py
│     │  │  │     │  │  ├─ eval_ckpt_main_tf1.py
│     │  │  │     │  │  ├─ preprocessing.py
│     │  │  │     │  │  └─ utils.py
│     │  │  │     │  ├─ rename.sh
│     │  │  │     │  └─ run.sh
│     │  │  │     └─ pretrained_tensorflow
│     │  │  │        └─ download.sh
│     │  │  ├─ __init__.py
│     │  │  ├─ __pycache__
│     │  │  │  ├─ __init__.cpython-36.pyc
│     │  │  │  ├─ _base.cpython-36.pyc
│     │  │  │  ├─ _preprocessing.cpython-36.pyc
│     │  │  │  ├─ _utils.cpython-36.pyc
│     │  │  │  ├─ densenet.cpython-36.pyc
│     │  │  │  ├─ dpn.cpython-36.pyc
│     │  │  │  ├─ efficientnet.cpython-36.pyc
│     │  │  │  ├─ inceptionresnetv2.cpython-36.pyc
│     │  │  │  ├─ inceptionv4.cpython-36.pyc
│     │  │  │  ├─ mobilenet.cpython-36.pyc
│     │  │  │  ├─ resnet.cpython-36.pyc
│     │  │  │  ├─ senet.cpython-36.pyc
│     │  │  │  ├─ vgg.cpython-36.pyc
│     │  │  │  └─ xception.cpython-36.pyc
│     │  │  ├─ _base.py
│     │  │  ├─ _preprocessing.py
│     │  │  ├─ _utils.py
│     │  │  ├─ efficientnet.py
│     │  │  └─ efficientnet_pytorch
│     │  │     ├─ __init__.py
│     │  │     ├─ __pycache__
│     │  │     │  ├─ __init__.cpython-36.pyc
│     │  │     │  ├─ model.cpython-36.pyc
│     │  │     │  └─ utils.cpython-36.pyc
│     │  │     ├─ model.py
│     │  │     └─ utils.py
│     │  └─ fpn
│     │     ├─ __init__.py
│     │     ├─ __pycache__
│     │     │  ├─ __init__.cpython-36.pyc
│     │     │  ├─ decoder.cpython-36.pyc
│     │     │  └─ model.cpython-36.pyc
│     │     ├─ decoder.py
│     │     └─ model.py
│     ├─ utils.py
│     └─ views.py
├─ config.py
├─ darknet
│  ├─ 3rdparty
│  │  ├─ pthreads
│  │  │  ├─ bin
│  │  │  │  ├─ pthreadGC2.dll
│  │  │  │  └─ pthreadVC2.dll
│  │  │  ├─ include
│  │  │  │  ├─ pthread.h
│  │  │  │  ├─ sched.h
│  │  │  │  └─ semaphore.h
│  │  │  └─ lib
│  │  │     ├─ libpthreadGC2.a
│  │  │     └─ pthreadVC2.lib
│  │  └─ stb
│  │     └─ include
│  │        ├─ stb_image.h
│  │        └─ stb_image_write.h
│  ├─ CMakeLists.txt
│  ├─ DarknetConfig.cmake.in
│  ├─ LICENSE
│  ├─ Makefile
│  ├─ README.md
│  ├─ backup
│  ├─ build
│  │  └─ darknet
│  │     ├─ YoloWrapper.cs
│  │     ├─ darknet.sln
│  │     ├─ darknet.vcxproj
│  │     ├─ darknet_no_gpu.sln
│  │     ├─ darknet_no_gpu.vcxproj
│  │     ├─ x64
│  │     │  ├─ calc_anchors.cmd
│  │     │  ├─ calc_mAP.cmd
│  │     │  ├─ calc_mAP_coco.cmd
│  │     │  ├─ calc_mAP_voc_py.cmd
│  │     │  ├─ cfg
│  │     │  │  ├─ Gaussian_yolov3_BDD.cfg
│  │     │  │  ├─ alexnet.cfg
│  │     │  │  ├─ cd53paspp-gamma.cfg
│  │     │  │  ├─ cifar.cfg
│  │     │  │  ├─ cifar.test.cfg
│  │     │  │  ├─ coco.data
│  │     │  │  ├─ combine9k.data
│  │     │  │  ├─ crnn.train.cfg
│  │     │  │  ├─ csdarknet53-omega.cfg
│  │     │  │  ├─ cspx-p7-mish-omega.cfg
│  │     │  │  ├─ cspx-p7-mish_hp.cfg
│  │     │  │  ├─ csresnext50-panet-spp-original-optimal.cfg
│  │     │  │  ├─ csresnext50-panet-spp.cfg
│  │     │  │  ├─ darknet.cfg
│  │     │  │  ├─ darknet19.cfg
│  │     │  │  ├─ darknet19_448.cfg
│  │     │  │  ├─ darknet53.cfg
│  │     │  │  ├─ darknet53_448_xnor.cfg
│  │     │  │  ├─ densenet201.cfg
│  │     │  │  ├─ efficientnet-lite3.cfg
│  │     │  │  ├─ efficientnet_b0.cfg
│  │     │  │  ├─ enet-coco.cfg
│  │     │  │  ├─ extraction.cfg
│  │     │  │  ├─ extraction.conv.cfg
│  │     │  │  ├─ extraction22k.cfg
│  │     │  │  ├─ go.test.cfg
│  │     │  │  ├─ gru.cfg
│  │     │  │  ├─ imagenet1k.data
│  │     │  │  ├─ imagenet22k.dataset
│  │     │  │  ├─ imagenet9k.hierarchy.dataset
│  │     │  │  ├─ jnet-conv.cfg
│  │     │  │  ├─ lstm.train.cfg
│  │     │  │  ├─ openimages.data
│  │     │  │  ├─ resnet101.cfg
│  │     │  │  ├─ resnet152.cfg
│  │     │  │  ├─ resnet152_trident.cfg
│  │     │  │  ├─ resnet50.cfg
│  │     │  │  ├─ resnext152-32x4d.cfg
│  │     │  │  ├─ rnn.cfg
│  │     │  │  ├─ rnn.train.cfg
│  │     │  │  ├─ strided.cfg
│  │     │  │  ├─ t1.test.cfg
│  │     │  │  ├─ tiny-yolo-voc.cfg
│  │     │  │  ├─ tiny-yolo.cfg
│  │     │  │  ├─ tiny-yolo_xnor.cfg
│  │     │  │  ├─ tiny.cfg
│  │     │  │  ├─ vgg-16.cfg
│  │     │  │  ├─ vgg-conv.cfg
│  │     │  │  ├─ voc.data
│  │     │  │  ├─ writing.cfg
│  │     │  │  ├─ yolo-voc.2.0.cfg
│  │     │  │  ├─ yolo-voc.cfg
│  │     │  │  ├─ yolo.2.0.cfg
│  │     │  │  ├─ yolo.cfg
│  │     │  │  ├─ yolo9000.cfg
│  │     │  │  ├─ yolov2-tiny-voc.cfg
│  │     │  │  ├─ yolov2-tiny.cfg
│  │     │  │  ├─ yolov2-voc.cfg
│  │     │  │  ├─ yolov2.cfg
│  │     │  │  ├─ yolov3-openimages.cfg
│  │     │  │  ├─ yolov3-spp.cfg
│  │     │  │  ├─ yolov3-tiny-prn.cfg
│  │     │  │  ├─ yolov3-tiny.cfg
│  │     │  │  ├─ yolov3-tiny_3l.cfg
│  │     │  │  ├─ yolov3-tiny_obj.cfg
│  │     │  │  ├─ yolov3-tiny_occlusion_track.cfg
│  │     │  │  ├─ yolov3-tiny_xnor.cfg
│  │     │  │  ├─ yolov3-voc.cfg
│  │     │  │  ├─ yolov3-voc.yolov3-giou-40.cfg
│  │     │  │  ├─ yolov3.cfg
│  │     │  │  ├─ yolov3.coco-giou-12.cfg
│  │     │  │  ├─ yolov3_5l.cfg
│  │     │  │  ├─ yolov4-custom.cfg
│  │     │  │  ├─ yolov4-tiny-3l.cfg
│  │     │  │  ├─ yolov4-tiny-custom.cfg
│  │     │  │  ├─ yolov4-tiny.cfg
│  │     │  │  ├─ yolov4-tiny_contrastive.cfg
│  │     │  │  └─ yolov4.cfg
│  │     │  ├─ classifier_densenet201.cmd
│  │     │  ├─ classifier_resnet50.cmd
│  │     │  ├─ darknet.py
│  │     │  ├─ darknet_coco.cmd
│  │     │  ├─ darknet_coco_9000.cmd
│  │     │  ├─ darknet_coco_9000_demo.cmd
│  │     │  ├─ darknet_demo_coco.cmd
│  │     │  ├─ darknet_demo_json_stream.cmd
│  │     │  ├─ darknet_demo_mjpeg_stream.cmd
│  │     │  ├─ darknet_demo_store.cmd
│  │     │  ├─ darknet_demo_voc.cmd
│  │     │  ├─ darknet_demo_voc_param.cmd
│  │     │  ├─ darknet_demo_voc_tiny.cmd
│  │     │  ├─ darknet_json_reslut.cmd
│  │     │  ├─ darknet_many_images.cmd
│  │     │  ├─ darknet_net_cam_coco.cmd
│  │     │  ├─ darknet_net_cam_voc.cmd
│  │     │  ├─ darknet_python.cmd
│  │     │  ├─ darknet_tiny_v2.cmd
│  │     │  ├─ darknet_video.cmd
│  │     │  ├─ darknet_video.py
│  │     │  ├─ darknet_voc.cmd
│  │     │  ├─ darknet_voc_tiny_v2.cmd
│  │     │  ├─ darknet_web_cam_voc.cmd
│  │     │  ├─ darknet_yolo_v3.cmd
│  │     │  ├─ darknet_yolo_v3_openimages.cmd
│  │     │  ├─ darknet_yolo_v3_video.cmd
│  │     │  ├─ darknet_yolov3_pseudo_labeling.cmd
│  │     │  ├─ data
│  │     │  │  ├─ 9k.labels
│  │     │  │  ├─ 9k.names
│  │     │  │  ├─ 9k.tree
│  │     │  │  ├─ coco.data
│  │     │  │  ├─ coco.names
│  │     │  │  ├─ coco9k.map
│  │     │  │  ├─ combine9k.data
│  │     │  │  ├─ dog.jpg
│  │     │  │  ├─ eagle.jpg
│  │     │  │  ├─ giraffe.jpg
│  │     │  │  ├─ goal.txt
│  │     │  │  ├─ horses.jpg
│  │     │  │  ├─ imagenet.labels.list
│  │     │  │  ├─ imagenet.shortnames.list
│  │     │  │  ├─ inet9k.map
│  │     │  │  ├─ labels
│  │     │  │  │  ├─ 100_0.png
│  │     │  │  │  ├─ 100_1.png
│  │     │  │  │  ├─ 100_2.png
│  │     │  │  │  ├─ 100_3.png
│  │     │  │  │  ├─ 100_4.png
│  │     │  │  │  ├─ 100_5.png
│  │     │  │  │  ├─ 100_6.png
│  │     │  │  │  ├─ 100_7.png
│  │     │  │  │  ├─ 101_0.png
│  │     │  │  │  ├─ 101_1.png
│  │     │  │  │  ├─ 101_2.png
│  │     │  │  │  ├─ 101_3.png
│  │     │  │  │  ├─ 101_4.png
│  │     │  │  │  ├─ 101_5.png
│  │     │  │  │  ├─ 101_6.png
│  │     │  │  │  ├─ 101_7.png
│  │     │  │  │  ├─ 102_0.png
│  │     │  │  │  ├─ 102_1.png
│  │     │  │  │  ├─ 102_2.png
│  │     │  │  │  ├─ 102_3.png
│  │     │  │  │  ├─ 102_4.png
│  │     │  │  │  ├─ 102_5.png
│  │     │  │  │  ├─ 102_6.png
│  │     │  │  │  ├─ 102_7.png
│  │     │  │  │  ├─ 103_0.png
│  │     │  │  │  ├─ 103_1.png
│  │     │  │  │  ├─ 103_2.png
│  │     │  │  │  ├─ 103_3.png
│  │     │  │  │  ├─ 103_4.png
│  │     │  │  │  ├─ 103_5.png
│  │     │  │  │  ├─ 103_6.png
│  │     │  │  │  ├─ 103_7.png
│  │     │  │  │  ├─ 104_0.png
│  │     │  │  │  ├─ 104_1.png
│  │     │  │  │  ├─ 104_2.png
│  │     │  │  │  ├─ 104_3.png
│  │     │  │  │  ├─ 104_4.png
│  │     │  │  │  ├─ 104_5.png
│  │     │  │  │  ├─ 104_6.png
│  │     │  │  │  ├─ 104_7.png
│  │     │  │  │  ├─ 105_0.png
│  │     │  │  │  ├─ 105_1.png
│  │     │  │  │  ├─ 105_2.png
│  │     │  │  │  ├─ 105_3.png
│  │     │  │  │  ├─ 105_4.png
│  │     │  │  │  ├─ 105_5.png
│  │     │  │  │  ├─ 105_6.png
│  │     │  │  │  ├─ 105_7.png
│  │     │  │  │  ├─ 106_0.png
│  │     │  │  │  ├─ 106_1.png
│  │     │  │  │  ├─ 106_2.png
│  │     │  │  │  ├─ 106_3.png
│  │     │  │  │  ├─ 106_4.png
│  │     │  │  │  ├─ 106_5.png
│  │     │  │  │  ├─ 106_6.png
│  │     │  │  │  ├─ 106_7.png
│  │     │  │  │  ├─ 107_0.png
│  │     │  │  │  ├─ 107_1.png
│  │     │  │  │  ├─ 107_2.png
│  │     │  │  │  ├─ 107_3.png
│  │     │  │  │  ├─ 107_4.png
│  │     │  │  │  ├─ 107_5.png
│  │     │  │  │  ├─ 107_6.png
│  │     │  │  │  ├─ 107_7.png
│  │     │  │  │  ├─ 108_0.png
│  │     │  │  │  ├─ 108_1.png
│  │     │  │  │  ├─ 108_2.png
│  │     │  │  │  ├─ 108_3.png
│  │     │  │  │  ├─ 108_4.png
│  │     │  │  │  ├─ 108_5.png
│  │     │  │  │  ├─ 108_6.png
│  │     │  │  │  ├─ 108_7.png
│  │     │  │  │  ├─ 109_0.png
│  │     │  │  │  ├─ 109_1.png
│  │     │  │  │  ├─ 109_2.png
│  │     │  │  │  ├─ 109_3.png
│  │     │  │  │  ├─ 109_4.png
│  │     │  │  │  ├─ 109_5.png
│  │     │  │  │  ├─ 109_6.png
│  │     │  │  │  ├─ 109_7.png
│  │     │  │  │  ├─ 110_0.png
│  │     │  │  │  ├─ 110_1.png
│  │     │  │  │  ├─ 110_2.png
│  │     │  │  │  ├─ 110_3.png
│  │     │  │  │  ├─ 110_4.png
│  │     │  │  │  ├─ 110_5.png
│  │     │  │  │  ├─ 110_6.png
│  │     │  │  │  ├─ 110_7.png
│  │     │  │  │  ├─ 111_0.png
│  │     │  │  │  ├─ 111_1.png
│  │     │  │  │  ├─ 111_2.png
│  │     │  │  │  ├─ 111_3.png
│  │     │  │  │  ├─ 111_4.png
│  │     │  │  │  ├─ 111_5.png
│  │     │  │  │  ├─ 111_6.png
│  │     │  │  │  ├─ 111_7.png
│  │     │  │  │  ├─ 112_0.png
│  │     │  │  │  ├─ 112_1.png
│  │     │  │  │  ├─ 112_2.png
│  │     │  │  │  ├─ 112_3.png
│  │     │  │  │  ├─ 112_4.png
│  │     │  │  │  ├─ 112_5.png
│  │     │  │  │  ├─ 112_6.png
│  │     │  │  │  ├─ 112_7.png
│  │     │  │  │  ├─ 113_0.png
│  │     │  │  │  ├─ 113_1.png
│  │     │  │  │  ├─ 113_2.png
│  │     │  │  │  ├─ 113_3.png
│  │     │  │  │  ├─ 113_4.png
│  │     │  │  │  ├─ 113_5.png
│  │     │  │  │  ├─ 113_6.png
│  │     │  │  │  ├─ 113_7.png
│  │     │  │  │  ├─ 114_0.png
│  │     │  │  │  ├─ 114_1.png
│  │     │  │  │  ├─ 114_2.png
│  │     │  │  │  ├─ 114_3.png
│  │     │  │  │  ├─ 114_4.png
│  │     │  │  │  ├─ 114_5.png
│  │     │  │  │  ├─ 114_6.png
│  │     │  │  │  ├─ 114_7.png
│  │     │  │  │  ├─ 115_0.png
│  │     │  │  │  ├─ 115_1.png
│  │     │  │  │  ├─ 115_2.png
│  │     │  │  │  ├─ 115_3.png
│  │     │  │  │  ├─ 115_4.png
│  │     │  │  │  ├─ 115_5.png
│  │     │  │  │  ├─ 115_6.png
│  │     │  │  │  ├─ 115_7.png
│  │     │  │  │  ├─ 116_0.png
│  │     │  │  │  ├─ 116_1.png
│  │     │  │  │  ├─ 116_2.png
│  │     │  │  │  ├─ 116_3.png
│  │     │  │  │  ├─ 116_4.png
│  │     │  │  │  ├─ 116_5.png
│  │     │  │  │  ├─ 116_6.png
│  │     │  │  │  ├─ 116_7.png
│  │     │  │  │  ├─ 117_0.png
│  │     │  │  │  ├─ 117_1.png
│  │     │  │  │  ├─ 117_2.png
│  │     │  │  │  ├─ 117_3.png
│  │     │  │  │  ├─ 117_4.png
│  │     │  │  │  ├─ 117_5.png
│  │     │  │  │  ├─ 117_6.png
│  │     │  │  │  ├─ 117_7.png
│  │     │  │  │  ├─ 118_0.png
│  │     │  │  │  ├─ 118_1.png
│  │     │  │  │  ├─ 118_2.png
│  │     │  │  │  ├─ 118_3.png
│  │     │  │  │  ├─ 118_4.png
│  │     │  │  │  ├─ 118_5.png
│  │     │  │  │  ├─ 118_6.png
│  │     │  │  │  ├─ 118_7.png
│  │     │  │  │  ├─ 119_0.png
│  │     │  │  │  ├─ 119_1.png
│  │     │  │  │  ├─ 119_2.png
│  │     │  │  │  ├─ 119_3.png
│  │     │  │  │  ├─ 119_4.png
│  │     │  │  │  ├─ 119_5.png
│  │     │  │  │  ├─ 119_6.png
│  │     │  │  │  ├─ 119_7.png
│  │     │  │  │  ├─ 120_0.png
│  │     │  │  │  ├─ 120_1.png
│  │     │  │  │  ├─ 120_2.png
│  │     │  │  │  ├─ 120_3.png
│  │     │  │  │  ├─ 120_4.png
│  │     │  │  │  ├─ 120_5.png
│  │     │  │  │  ├─ 120_6.png
│  │     │  │  │  ├─ 120_7.png
│  │     │  │  │  ├─ 121_0.png
│  │     │  │  │  ├─ 121_1.png
│  │     │  │  │  ├─ 121_2.png
│  │     │  │  │  ├─ 121_3.png
│  │     │  │  │  ├─ 121_4.png
│  │     │  │  │  ├─ 121_5.png
│  │     │  │  │  ├─ 121_6.png
│  │     │  │  │  ├─ 121_7.png
│  │     │  │  │  ├─ 122_0.png
│  │     │  │  │  ├─ 122_1.png
│  │     │  │  │  ├─ 122_2.png
│  │     │  │  │  ├─ 122_3.png
│  │     │  │  │  ├─ 122_4.png
│  │     │  │  │  ├─ 122_5.png
│  │     │  │  │  ├─ 122_6.png
│  │     │  │  │  ├─ 122_7.png
│  │     │  │  │  ├─ 123_0.png
│  │     │  │  │  ├─ 123_1.png
│  │     │  │  │  ├─ 123_2.png
│  │     │  │  │  ├─ 123_3.png
│  │     │  │  │  ├─ 123_4.png
│  │     │  │  │  ├─ 123_5.png
│  │     │  │  │  ├─ 123_6.png
│  │     │  │  │  ├─ 123_7.png
│  │     │  │  │  ├─ 124_0.png
│  │     │  │  │  ├─ 124_1.png
│  │     │  │  │  ├─ 124_2.png
│  │     │  │  │  ├─ 124_3.png
│  │     │  │  │  ├─ 124_4.png
│  │     │  │  │  ├─ 124_5.png
│  │     │  │  │  ├─ 124_6.png
│  │     │  │  │  ├─ 124_7.png
│  │     │  │  │  ├─ 125_0.png
│  │     │  │  │  ├─ 125_1.png
│  │     │  │  │  ├─ 125_2.png
│  │     │  │  │  ├─ 125_3.png
│  │     │  │  │  ├─ 125_4.png
│  │     │  │  │  ├─ 125_5.png
│  │     │  │  │  ├─ 125_6.png
│  │     │  │  │  ├─ 125_7.png
│  │     │  │  │  ├─ 126_0.png
│  │     │  │  │  ├─ 126_1.png
│  │     │  │  │  ├─ 126_2.png
│  │     │  │  │  ├─ 126_3.png
│  │     │  │  │  ├─ 126_4.png
│  │     │  │  │  ├─ 126_5.png
│  │     │  │  │  ├─ 126_6.png
│  │     │  │  │  ├─ 126_7.png
│  │     │  │  │  ├─ 32_0.png
│  │     │  │  │  ├─ 32_1.png
│  │     │  │  │  ├─ 32_2.png
│  │     │  │  │  ├─ 32_3.png
│  │     │  │  │  ├─ 32_4.png
│  │     │  │  │  ├─ 32_5.png
│  │     │  │  │  ├─ 32_6.png
│  │     │  │  │  ├─ 32_7.png
│  │     │  │  │  ├─ 33_0.png
│  │     │  │  │  ├─ 33_1.png
│  │     │  │  │  ├─ 33_2.png
│  │     │  │  │  ├─ 33_3.png
│  │     │  │  │  ├─ 33_4.png
│  │     │  │  │  ├─ 33_5.png
│  │     │  │  │  ├─ 33_6.png
│  │     │  │  │  ├─ 33_7.png
│  │     │  │  │  ├─ 34_0.png
│  │     │  │  │  ├─ 34_1.png
│  │     │  │  │  ├─ 34_2.png
│  │     │  │  │  ├─ 34_3.png
│  │     │  │  │  ├─ 34_4.png
│  │     │  │  │  ├─ 34_5.png
│  │     │  │  │  ├─ 34_6.png
│  │     │  │  │  ├─ 34_7.png
│  │     │  │  │  ├─ 35_0.png
│  │     │  │  │  ├─ 35_1.png
│  │     │  │  │  ├─ 35_2.png
│  │     │  │  │  ├─ 35_3.png
│  │     │  │  │  ├─ 35_4.png
│  │     │  │  │  ├─ 35_5.png
│  │     │  │  │  ├─ 35_6.png
│  │     │  │  │  ├─ 35_7.png
│  │     │  │  │  ├─ 36_0.png
│  │     │  │  │  ├─ 36_1.png
│  │     │  │  │  ├─ 36_2.png
│  │     │  │  │  ├─ 36_3.png
│  │     │  │  │  ├─ 36_4.png
│  │     │  │  │  ├─ 36_5.png
│  │     │  │  │  ├─ 36_6.png
│  │     │  │  │  ├─ 36_7.png
│  │     │  │  │  ├─ 37_0.png
│  │     │  │  │  ├─ 37_1.png
│  │     │  │  │  ├─ 37_2.png
│  │     │  │  │  ├─ 37_3.png
│  │     │  │  │  ├─ 37_4.png
│  │     │  │  │  ├─ 37_5.png
│  │     │  │  │  ├─ 37_6.png
│  │     │  │  │  ├─ 37_7.png
│  │     │  │  │  ├─ 38_0.png
│  │     │  │  │  ├─ 38_1.png
│  │     │  │  │  ├─ 38_2.png
│  │     │  │  │  ├─ 38_3.png
│  │     │  │  │  ├─ 38_4.png
│  │     │  │  │  ├─ 38_5.png
│  │     │  │  │  ├─ 38_6.png
│  │     │  │  │  ├─ 38_7.png
│  │     │  │  │  ├─ 39_0.png
│  │     │  │  │  ├─ 39_1.png
│  │     │  │  │  ├─ 39_2.png
│  │     │  │  │  ├─ 39_3.png
│  │     │  │  │  ├─ 39_4.png
│  │     │  │  │  ├─ 39_5.png
│  │     │  │  │  ├─ 39_6.png
│  │     │  │  │  ├─ 39_7.png
│  │     │  │  │  ├─ 40_0.png
│  │     │  │  │  ├─ 40_1.png
│  │     │  │  │  ├─ 40_2.png
│  │     │  │  │  ├─ 40_3.png
│  │     │  │  │  ├─ 40_4.png
│  │     │  │  │  ├─ 40_5.png
│  │     │  │  │  ├─ 40_6.png
│  │     │  │  │  ├─ 40_7.png
│  │     │  │  │  ├─ 41_0.png
│  │     │  │  │  ├─ 41_1.png
│  │     │  │  │  ├─ 41_2.png
│  │     │  │  │  ├─ 41_3.png
│  │     │  │  │  ├─ 41_4.png
│  │     │  │  │  ├─ 41_5.png
│  │     │  │  │  ├─ 41_6.png
│  │     │  │  │  ├─ 41_7.png
│  │     │  │  │  ├─ 42_0.png
│  │     │  │  │  ├─ 42_1.png
│  │     │  │  │  ├─ 42_2.png
│  │     │  │  │  ├─ 42_3.png
│  │     │  │  │  ├─ 42_4.png
│  │     │  │  │  ├─ 42_5.png
│  │     │  │  │  ├─ 42_6.png
│  │     │  │  │  ├─ 42_7.png
│  │     │  │  │  ├─ 43_0.png
│  │     │  │  │  ├─ 43_1.png
│  │     │  │  │  ├─ 43_2.png
│  │     │  │  │  ├─ 43_3.png
│  │     │  │  │  ├─ 43_4.png
│  │     │  │  │  ├─ 43_5.png
│  │     │  │  │  ├─ 43_6.png
│  │     │  │  │  ├─ 43_7.png
│  │     │  │  │  ├─ 44_0.png
│  │     │  │  │  ├─ 44_1.png
│  │     │  │  │  ├─ 44_2.png
│  │     │  │  │  ├─ 44_3.png
│  │     │  │  │  ├─ 44_4.png
│  │     │  │  │  ├─ 44_5.png
│  │     │  │  │  ├─ 44_6.png
│  │     │  │  │  ├─ 44_7.png
│  │     │  │  │  ├─ 45_0.png
│  │     │  │  │  ├─ 45_1.png
│  │     │  │  │  ├─ 45_2.png
│  │     │  │  │  ├─ 45_3.png
│  │     │  │  │  ├─ 45_4.png
│  │     │  │  │  ├─ 45_5.png
│  │     │  │  │  ├─ 45_6.png
│  │     │  │  │  ├─ 45_7.png
│  │     │  │  │  ├─ 46_0.png
│  │     │  │  │  ├─ 46_1.png
│  │     │  │  │  ├─ 46_2.png
│  │     │  │  │  ├─ 46_3.png
│  │     │  │  │  ├─ 46_4.png
│  │     │  │  │  ├─ 46_5.png
│  │     │  │  │  ├─ 46_6.png
│  │     │  │  │  ├─ 46_7.png
│  │     │  │  │  ├─ 47_0.png
│  │     │  │  │  ├─ 47_1.png
│  │     │  │  │  ├─ 47_2.png
│  │     │  │  │  ├─ 47_3.png
│  │     │  │  │  ├─ 47_4.png
│  │     │  │  │  ├─ 47_5.png
│  │     │  │  │  ├─ 47_6.png
│  │     │  │  │  ├─ 47_7.png
│  │     │  │  │  ├─ 48_0.png
│  │     │  │  │  ├─ 48_1.png
│  │     │  │  │  ├─ 48_2.png
│  │     │  │  │  ├─ 48_3.png
│  │     │  │  │  ├─ 48_4.png
│  │     │  │  │  ├─ 48_5.png
│  │     │  │  │  ├─ 48_6.png
│  │     │  │  │  ├─ 48_7.png
│  │     │  │  │  ├─ 49_0.png
│  │     │  │  │  ├─ 49_1.png
│  │     │  │  │  ├─ 49_2.png
│  │     │  │  │  ├─ 49_3.png
│  │     │  │  │  ├─ 49_4.png
│  │     │  │  │  ├─ 49_5.png
│  │     │  │  │  ├─ 49_6.png
│  │     │  │  │  ├─ 49_7.png
│  │     │  │  │  ├─ 50_0.png
│  │     │  │  │  ├─ 50_1.png
│  │     │  │  │  ├─ 50_2.png
│  │     │  │  │  ├─ 50_3.png
│  │     │  │  │  ├─ 50_4.png
│  │     │  │  │  ├─ 50_5.png
│  │     │  │  │  ├─ 50_6.png
│  │     │  │  │  ├─ 50_7.png
│  │     │  │  │  ├─ 51_0.png
│  │     │  │  │  ├─ 51_1.png
│  │     │  │  │  ├─ 51_2.png
│  │     │  │  │  ├─ 51_3.png
│  │     │  │  │  ├─ 51_4.png
│  │     │  │  │  ├─ 51_5.png
│  │     │  │  │  ├─ 51_6.png
│  │     │  │  │  ├─ 51_7.png
│  │     │  │  │  ├─ 52_0.png
│  │     │  │  │  ├─ 52_1.png
│  │     │  │  │  ├─ 52_2.png
│  │     │  │  │  ├─ 52_3.png
│  │     │  │  │  ├─ 52_4.png
│  │     │  │  │  ├─ 52_5.png
│  │     │  │  │  ├─ 52_6.png
│  │     │  │  │  ├─ 52_7.png
│  │     │  │  │  ├─ 53_0.png
│  │     │  │  │  ├─ 53_1.png
│  │     │  │  │  ├─ 53_2.png
│  │     │  │  │  ├─ 53_3.png
│  │     │  │  │  ├─ 53_4.png
│  │     │  │  │  ├─ 53_5.png
│  │     │  │  │  ├─ 53_6.png
│  │     │  │  │  ├─ 53_7.png
│  │     │  │  │  ├─ 54_0.png
│  │     │  │  │  ├─ 54_1.png
│  │     │  │  │  ├─ 54_2.png
│  │     │  │  │  ├─ 54_3.png
│  │     │  │  │  ├─ 54_4.png
│  │     │  │  │  ├─ 54_5.png
│  │     │  │  │  ├─ 54_6.png
│  │     │  │  │  ├─ 54_7.png
│  │     │  │  │  ├─ 55_0.png
│  │     │  │  │  ├─ 55_1.png
│  │     │  │  │  ├─ 55_2.png
│  │     │  │  │  ├─ 55_3.png
│  │     │  │  │  ├─ 55_4.png
│  │     │  │  │  ├─ 55_5.png
│  │     │  │  │  ├─ 55_6.png
│  │     │  │  │  ├─ 55_7.png
│  │     │  │  │  ├─ 56_0.png
│  │     │  │  │  ├─ 56_1.png
│  │     │  │  │  ├─ 56_2.png
│  │     │  │  │  ├─ 56_3.png
│  │     │  │  │  ├─ 56_4.png
│  │     │  │  │  ├─ 56_5.png
│  │     │  │  │  ├─ 56_6.png
│  │     │  │  │  ├─ 56_7.png
│  │     │  │  │  ├─ 57_0.png
│  │     │  │  │  ├─ 57_1.png
│  │     │  │  │  ├─ 57_2.png
│  │     │  │  │  ├─ 57_3.png
│  │     │  │  │  ├─ 57_4.png
│  │     │  │  │  ├─ 57_5.png
│  │     │  │  │  ├─ 57_6.png
│  │     │  │  │  ├─ 57_7.png
│  │     │  │  │  ├─ 58_0.png
│  │     │  │  │  ├─ 58_1.png
│  │     │  │  │  ├─ 58_2.png
│  │     │  │  │  ├─ 58_3.png
│  │     │  │  │  ├─ 58_4.png
│  │     │  │  │  ├─ 58_5.png
│  │     │  │  │  ├─ 58_6.png
│  │     │  │  │  ├─ 58_7.png
│  │     │  │  │  ├─ 59_0.png
│  │     │  │  │  ├─ 59_1.png
│  │     │  │  │  ├─ 59_2.png
│  │     │  │  │  ├─ 59_3.png
│  │     │  │  │  ├─ 59_4.png
│  │     │  │  │  ├─ 59_5.png
│  │     │  │  │  ├─ 59_6.png
│  │     │  │  │  ├─ 59_7.png
│  │     │  │  │  ├─ 60_0.png
│  │     │  │  │  ├─ 60_1.png
│  │     │  │  │  ├─ 60_2.png
│  │     │  │  │  ├─ 60_3.png
│  │     │  │  │  ├─ 60_4.png
│  │     │  │  │  ├─ 60_5.png
│  │     │  │  │  ├─ 60_6.png
│  │     │  │  │  ├─ 60_7.png
│  │     │  │  │  ├─ 61_0.png
│  │     │  │  │  ├─ 61_1.png
│  │     │  │  │  ├─ 61_2.png
│  │     │  │  │  ├─ 61_3.png
│  │     │  │  │  ├─ 61_4.png
│  │     │  │  │  ├─ 61_5.png
│  │     │  │  │  ├─ 61_6.png
│  │     │  │  │  ├─ 61_7.png
│  │     │  │  │  ├─ 62_0.png
│  │     │  │  │  ├─ 62_1.png
│  │     │  │  │  ├─ 62_2.png
│  │     │  │  │  ├─ 62_3.png
│  │     │  │  │  ├─ 62_4.png
│  │     │  │  │  ├─ 62_5.png
│  │     │  │  │  ├─ 62_6.png
│  │     │  │  │  ├─ 62_7.png
│  │     │  │  │  ├─ 63_0.png
│  │     │  │  │  ├─ 63_1.png
│  │     │  │  │  ├─ 63_2.png
│  │     │  │  │  ├─ 63_3.png
│  │     │  │  │  ├─ 63_4.png
│  │     │  │  │  ├─ 63_5.png
│  │     │  │  │  ├─ 63_6.png
│  │     │  │  │  ├─ 63_7.png
│  │     │  │  │  ├─ 64_0.png
│  │     │  │  │  ├─ 64_1.png
│  │     │  │  │  ├─ 64_2.png
│  │     │  │  │  ├─ 64_3.png
│  │     │  │  │  ├─ 64_4.png
│  │     │  │  │  ├─ 64_5.png
│  │     │  │  │  ├─ 64_6.png
│  │     │  │  │  ├─ 64_7.png
│  │     │  │  │  ├─ 65_0.png
│  │     │  │  │  ├─ 65_1.png
│  │     │  │  │  ├─ 65_2.png
│  │     │  │  │  ├─ 65_3.png
│  │     │  │  │  ├─ 65_4.png
│  │     │  │  │  ├─ 65_5.png
│  │     │  │  │  ├─ 65_6.png
│  │     │  │  │  ├─ 65_7.png
│  │     │  │  │  ├─ 66_0.png
│  │     │  │  │  ├─ 66_1.png
│  │     │  │  │  ├─ 66_2.png
│  │     │  │  │  ├─ 66_3.png
│  │     │  │  │  ├─ 66_4.png
│  │     │  │  │  ├─ 66_5.png
│  │     │  │  │  ├─ 66_6.png
│  │     │  │  │  ├─ 66_7.png
│  │     │  │  │  ├─ 67_0.png
│  │     │  │  │  ├─ 67_1.png
│  │     │  │  │  ├─ 67_2.png
│  │     │  │  │  ├─ 67_3.png
│  │     │  │  │  ├─ 67_4.png
│  │     │  │  │  ├─ 67_5.png
│  │     │  │  │  ├─ 67_6.png
│  │     │  │  │  ├─ 67_7.png
│  │     │  │  │  ├─ 68_0.png
│  │     │  │  │  ├─ 68_1.png
│  │     │  │  │  ├─ 68_2.png
│  │     │  │  │  ├─ 68_3.png
│  │     │  │  │  ├─ 68_4.png
│  │     │  │  │  ├─ 68_5.png
│  │     │  │  │  ├─ 68_6.png
│  │     │  │  │  ├─ 68_7.png
│  │     │  │  │  ├─ 69_0.png
│  │     │  │  │  ├─ 69_1.png
│  │     │  │  │  ├─ 69_2.png
│  │     │  │  │  ├─ 69_3.png
│  │     │  │  │  ├─ 69_4.png
│  │     │  │  │  ├─ 69_5.png
│  │     │  │  │  ├─ 69_6.png
│  │     │  │  │  ├─ 69_7.png
│  │     │  │  │  ├─ 70_0.png
│  │     │  │  │  ├─ 70_1.png
│  │     │  │  │  ├─ 70_2.png
│  │     │  │  │  ├─ 70_3.png
│  │     │  │  │  ├─ 70_4.png
│  │     │  │  │  ├─ 70_5.png
│  │     │  │  │  ├─ 70_6.png
│  │     │  │  │  ├─ 70_7.png
│  │     │  │  │  ├─ 71_0.png
│  │     │  │  │  ├─ 71_1.png
│  │     │  │  │  ├─ 71_2.png
│  │     │  │  │  ├─ 71_3.png
│  │     │  │  │  ├─ 71_4.png
│  │     │  │  │  ├─ 71_5.png
│  │     │  │  │  ├─ 71_6.png
│  │     │  │  │  ├─ 71_7.png
│  │     │  │  │  ├─ 72_0.png
│  │     │  │  │  ├─ 72_1.png
│  │     │  │  │  ├─ 72_2.png
│  │     │  │  │  ├─ 72_3.png
│  │     │  │  │  ├─ 72_4.png
│  │     │  │  │  ├─ 72_5.png
│  │     │  │  │  ├─ 72_6.png
│  │     │  │  │  ├─ 72_7.png
│  │     │  │  │  ├─ 73_0.png
│  │     │  │  │  ├─ 73_1.png
│  │     │  │  │  ├─ 73_2.png
│  │     │  │  │  ├─ 73_3.png
│  │     │  │  │  ├─ 73_4.png
│  │     │  │  │  ├─ 73_5.png
│  │     │  │  │  ├─ 73_6.png
│  │     │  │  │  ├─ 73_7.png
│  │     │  │  │  ├─ 74_0.png
│  │     │  │  │  ├─ 74_1.png
│  │     │  │  │  ├─ 74_2.png
│  │     │  │  │  ├─ 74_3.png
│  │     │  │  │  ├─ 74_4.png
│  │     │  │  │  ├─ 74_5.png
│  │     │  │  │  ├─ 74_6.png
│  │     │  │  │  ├─ 74_7.png
│  │     │  │  │  ├─ 75_0.png
│  │     │  │  │  ├─ 75_1.png
│  │     │  │  │  ├─ 75_2.png
│  │     │  │  │  ├─ 75_3.png
│  │     │  │  │  ├─ 75_4.png
│  │     │  │  │  ├─ 75_5.png
│  │     │  │  │  ├─ 75_6.png
│  │     │  │  │  ├─ 75_7.png
│  │     │  │  │  ├─ 76_0.png
│  │     │  │  │  ├─ 76_1.png
│  │     │  │  │  ├─ 76_2.png
│  │     │  │  │  ├─ 76_3.png
│  │     │  │  │  ├─ 76_4.png
│  │     │  │  │  ├─ 76_5.png
│  │     │  │  │  ├─ 76_6.png
│  │     │  │  │  ├─ 76_7.png
│  │     │  │  │  ├─ 77_0.png
│  │     │  │  │  ├─ 77_1.png
│  │     │  │  │  ├─ 77_2.png
│  │     │  │  │  ├─ 77_3.png
│  │     │  │  │  ├─ 77_4.png
│  │     │  │  │  ├─ 77_5.png
│  │     │  │  │  ├─ 77_6.png
│  │     │  │  │  ├─ 77_7.png
│  │     │  │  │  ├─ 78_0.png
│  │     │  │  │  ├─ 78_1.png
│  │     │  │  │  ├─ 78_2.png
│  │     │  │  │  ├─ 78_3.png
│  │     │  │  │  ├─ 78_4.png
│  │     │  │  │  ├─ 78_5.png
│  │     │  │  │  ├─ 78_6.png
│  │     │  │  │  ├─ 78_7.png
│  │     │  │  │  ├─ 79_0.png
│  │     │  │  │  ├─ 79_1.png
│  │     │  │  │  ├─ 79_2.png
│  │     │  │  │  ├─ 79_3.png
│  │     │  │  │  ├─ 79_4.png
│  │     │  │  │  ├─ 79_5.png
│  │     │  │  │  ├─ 79_6.png
│  │     │  │  │  ├─ 79_7.png
│  │     │  │  │  ├─ 80_0.png
│  │     │  │  │  ├─ 80_1.png
│  │     │  │  │  ├─ 80_2.png
│  │     │  │  │  ├─ 80_3.png
│  │     │  │  │  ├─ 80_4.png
│  │     │  │  │  ├─ 80_5.png
│  │     │  │  │  ├─ 80_6.png
│  │     │  │  │  ├─ 80_7.png
│  │     │  │  │  ├─ 81_0.png
│  │     │  │  │  ├─ 81_1.png
│  │     │  │  │  ├─ 81_2.png
│  │     │  │  │  ├─ 81_3.png
│  │     │  │  │  ├─ 81_4.png
│  │     │  │  │  ├─ 81_5.png
│  │     │  │  │  ├─ 81_6.png
│  │     │  │  │  ├─ 81_7.png
│  │     │  │  │  ├─ 82_0.png
│  │     │  │  │  ├─ 82_1.png
│  │     │  │  │  ├─ 82_2.png
│  │     │  │  │  ├─ 82_3.png
│  │     │  │  │  ├─ 82_4.png
│  │     │  │  │  ├─ 82_5.png
│  │     │  │  │  ├─ 82_6.png
│  │     │  │  │  ├─ 82_7.png
│  │     │  │  │  ├─ 83_0.png
│  │     │  │  │  ├─ 83_1.png
│  │     │  │  │  ├─ 83_2.png
│  │     │  │  │  ├─ 83_3.png
│  │     │  │  │  ├─ 83_4.png
│  │     │  │  │  ├─ 83_5.png
│  │     │  │  │  ├─ 83_6.png
│  │     │  │  │  ├─ 83_7.png
│  │     │  │  │  ├─ 84_0.png
│  │     │  │  │  ├─ 84_1.png
│  │     │  │  │  ├─ 84_2.png
│  │     │  │  │  ├─ 84_3.png
│  │     │  │  │  ├─ 84_4.png
│  │     │  │  │  ├─ 84_5.png
│  │     │  │  │  ├─ 84_6.png
│  │     │  │  │  ├─ 84_7.png
│  │     │  │  │  ├─ 85_0.png
│  │     │  │  │  ├─ 85_1.png
│  │     │  │  │  ├─ 85_2.png
│  │     │  │  │  ├─ 85_3.png
│  │     │  │  │  ├─ 85_4.png
│  │     │  │  │  ├─ 85_5.png
│  │     │  │  │  ├─ 85_6.png
│  │     │  │  │  ├─ 85_7.png
│  │     │  │  │  ├─ 86_0.png
│  │     │  │  │  ├─ 86_1.png
│  │     │  │  │  ├─ 86_2.png
│  │     │  │  │  ├─ 86_3.png
│  │     │  │  │  ├─ 86_4.png
│  │     │  │  │  ├─ 86_5.png
│  │     │  │  │  ├─ 86_6.png
│  │     │  │  │  ├─ 86_7.png
│  │     │  │  │  ├─ 87_0.png
│  │     │  │  │  ├─ 87_1.png
│  │     │  │  │  ├─ 87_2.png
│  │     │  │  │  ├─ 87_3.png
│  │     │  │  │  ├─ 87_4.png
│  │     │  │  │  ├─ 87_5.png
│  │     │  │  │  ├─ 87_6.png
│  │     │  │  │  ├─ 87_7.png
│  │     │  │  │  ├─ 88_0.png
│  │     │  │  │  ├─ 88_1.png
│  │     │  │  │  ├─ 88_2.png
│  │     │  │  │  ├─ 88_3.png
│  │     │  │  │  ├─ 88_4.png
│  │     │  │  │  ├─ 88_5.png
│  │     │  │  │  ├─ 88_6.png
│  │     │  │  │  ├─ 88_7.png
│  │     │  │  │  ├─ 89_0.png
│  │     │  │  │  ├─ 89_1.png
│  │     │  │  │  ├─ 89_2.png
│  │     │  │  │  ├─ 89_3.png
│  │     │  │  │  ├─ 89_4.png
│  │     │  │  │  ├─ 89_5.png
│  │     │  │  │  ├─ 89_6.png
│  │     │  │  │  ├─ 89_7.png
│  │     │  │  │  ├─ 90_0.png
│  │     │  │  │  ├─ 90_1.png
│  │     │  │  │  ├─ 90_2.png
│  │     │  │  │  ├─ 90_3.png
│  │     │  │  │  ├─ 90_4.png
│  │     │  │  │  ├─ 90_5.png
│  │     │  │  │  ├─ 90_6.png
│  │     │  │  │  ├─ 90_7.png
│  │     │  │  │  ├─ 91_0.png
│  │     │  │  │  ├─ 91_1.png
│  │     │  │  │  ├─ 91_2.png
│  │     │  │  │  ├─ 91_3.png
│  │     │  │  │  ├─ 91_4.png
│  │     │  │  │  ├─ 91_5.png
│  │     │  │  │  ├─ 91_6.png
│  │     │  │  │  ├─ 91_7.png
│  │     │  │  │  ├─ 92_0.png
│  │     │  │  │  ├─ 92_1.png
│  │     │  │  │  ├─ 92_2.png
│  │     │  │  │  ├─ 92_3.png
│  │     │  │  │  ├─ 92_4.png
│  │     │  │  │  ├─ 92_5.png
│  │     │  │  │  ├─ 92_6.png
│  │     │  │  │  ├─ 92_7.png
│  │     │  │  │  ├─ 93_0.png
│  │     │  │  │  ├─ 93_1.png
│  │     │  │  │  ├─ 93_2.png
│  │     │  │  │  ├─ 93_3.png
│  │     │  │  │  ├─ 93_4.png
│  │     │  │  │  ├─ 93_5.png
│  │     │  │  │  ├─ 93_6.png
│  │     │  │  │  ├─ 93_7.png
│  │     │  │  │  ├─ 94_0.png
│  │     │  │  │  ├─ 94_1.png
│  │     │  │  │  ├─ 94_2.png
│  │     │  │  │  ├─ 94_3.png
│  │     │  │  │  ├─ 94_4.png
│  │     │  │  │  ├─ 94_5.png
│  │     │  │  │  ├─ 94_6.png
│  │     │  │  │  ├─ 94_7.png
│  │     │  │  │  ├─ 95_0.png
│  │     │  │  │  ├─ 95_1.png
│  │     │  │  │  ├─ 95_2.png
│  │     │  │  │  ├─ 95_3.png
│  │     │  │  │  ├─ 95_4.png
│  │     │  │  │  ├─ 95_5.png
│  │     │  │  │  ├─ 95_6.png
│  │     │  │  │  ├─ 95_7.png
│  │     │  │  │  ├─ 96_0.png
│  │     │  │  │  ├─ 96_1.png
│  │     │  │  │  ├─ 96_2.png
│  │     │  │  │  ├─ 96_3.png
│  │     │  │  │  ├─ 96_4.png
│  │     │  │  │  ├─ 96_5.png
│  │     │  │  │  ├─ 96_6.png
│  │     │  │  │  ├─ 96_7.png
│  │     │  │  │  ├─ 97_0.png
│  │     │  │  │  ├─ 97_1.png
│  │     │  │  │  ├─ 97_2.png
│  │     │  │  │  ├─ 97_3.png
│  │     │  │  │  ├─ 97_4.png
│  │     │  │  │  ├─ 97_5.png
│  │     │  │  │  ├─ 97_6.png
│  │     │  │  │  ├─ 97_7.png
│  │     │  │  │  ├─ 98_0.png
│  │     │  │  │  ├─ 98_1.png
│  │     │  │  │  ├─ 98_2.png
│  │     │  │  │  ├─ 98_3.png
│  │     │  │  │  ├─ 98_4.png
│  │     │  │  │  ├─ 98_5.png
│  │     │  │  │  ├─ 98_6.png
│  │     │  │  │  ├─ 98_7.png
│  │     │  │  │  ├─ 99_0.png
│  │     │  │  │  ├─ 99_1.png
│  │     │  │  │  ├─ 99_2.png
│  │     │  │  │  ├─ 99_3.png
│  │     │  │  │  ├─ 99_4.png
│  │     │  │  │  ├─ 99_5.png
│  │     │  │  │  ├─ 99_6.png
│  │     │  │  │  ├─ 99_7.png
│  │     │  │  │  └─ make_labels.py
│  │     │  │  ├─ openimages.data
│  │     │  │  ├─ openimages.names
│  │     │  │  ├─ person.jpg
│  │     │  │  ├─ scream.jpg
│  │     │  │  ├─ voc
│  │     │  │  │  └─ voc_label.py
│  │     │  │  ├─ voc.data
│  │     │  │  └─ voc.names
│  │     │  ├─ densenet201_yolo.cfg
│  │     │  ├─ dog.jpg
│  │     │  ├─ dogr.jpg
│  │     │  ├─ gen_anchors.py
│  │     │  ├─ partial.cmd
│  │     │  ├─ pthreadGC2.dll
│  │     │  ├─ pthreadVC2.dll
│  │     │  ├─ resnet152_yolo.cfg
│  │     │  ├─ resnet50_yolo.cfg
│  │     │  ├─ reval_voc_py3.py
│  │     │  ├─ rnn_lstm.cmd
│  │     │  ├─ rnn_tolstoy.cmd
│  │     │  ├─ tiny-yolo-voc.cfg
│  │     │  ├─ tiny-yolo.cfg
│  │     │  ├─ train_voc.cmd
│  │     │  ├─ voc_eval_py3.py
│  │     │  ├─ yolo-voc.2.0.cfg
│  │     │  ├─ yolo-voc.cfg
│  │     │  ├─ yolo.2.0.cfg
│  │     │  ├─ yolo.cfg
│  │     │  ├─ yolo9000.cfg
│  │     │  ├─ yolov3-voc.cfg
│  │     │  └─ yolov3.cfg
│  │     ├─ yolo_console_dll.sln
│  │     ├─ yolo_console_dll.vcxproj
│  │     ├─ yolo_cpp_dll.sln
│  │     ├─ yolo_cpp_dll.vcxproj
│  │     ├─ yolo_cpp_dll_no_gpu.sln
│  │     └─ yolo_cpp_dll_no_gpu.vcxproj
│  ├─ build.ps1
│  ├─ build.sh
│  ├─ cfg
│  │  ├─ 9k.labels
│  │  ├─ 9k.names
│  │  ├─ 9k.tree
│  │  ├─ Gaussian_yolov3_BDD.cfg
│  │  ├─ alexnet.cfg
│  │  ├─ cd53paspp-gamma.cfg
│  │  ├─ cifar.cfg
│  │  ├─ cifar.test.cfg
│  │  ├─ coco.data
│  │  ├─ coco.names
│  │  ├─ coco9k.map
│  │  ├─ combine9k.data
│  │  ├─ crnn.train.cfg
│  │  ├─ csdarknet53-omega.cfg
│  │  ├─ cspx-p7-mish-omega.cfg
│  │  ├─ cspx-p7-mish_hp.cfg
│  │  ├─ csresnext50-panet-spp-original-optimal.cfg
│  │  ├─ csresnext50-panet-spp.cfg
│  │  ├─ darknet.cfg
│  │  ├─ darknet19.cfg
│  │  ├─ darknet19_448.cfg
│  │  ├─ darknet53.cfg
│  │  ├─ darknet53_448_xnor.cfg
│  │  ├─ densenet201.cfg
│  │  ├─ efficientnet-lite3.cfg
│  │  ├─ efficientnet_b0.cfg
│  │  ├─ enet-coco.cfg
│  │  ├─ extraction.cfg
│  │  ├─ extraction.conv.cfg
│  │  ├─ extraction22k.cfg
│  │  ├─ go.test.cfg
│  │  ├─ gru.cfg
│  │  ├─ imagenet.labels.list
│  │  ├─ imagenet.shortnames.list
│  │  ├─ imagenet1k.data
│  │  ├─ imagenet22k.dataset
│  │  ├─ imagenet9k.hierarchy.dataset
│  │  ├─ inet9k.map
│  │  ├─ jnet-conv.cfg
│  │  ├─ lstm.train.cfg
│  │  ├─ openimages.data
│  │  ├─ resnet101.cfg
│  │  ├─ resnet152.cfg
│  │  ├─ resnet152_trident.cfg
│  │  ├─ resnet50.cfg
│  │  ├─ resnext152-32x4d.cfg
│  │  ├─ rnn.cfg
│  │  ├─ rnn.train.cfg
│  │  ├─ strided.cfg
│  │  ├─ t1.test.cfg
│  │  ├─ tiny-yolo-voc.cfg
│  │  ├─ tiny-yolo.cfg
│  │  ├─ tiny-yolo_xnor.cfg
│  │  ├─ tiny.cfg
│  │  ├─ vgg-16.cfg
│  │  ├─ vgg-conv.cfg
│  │  ├─ voc.data
│  │  ├─ writing.cfg
│  │  ├─ yolo-voc.2.0.cfg
│  │  ├─ yolo-voc.cfg
│  │  ├─ yolo.2.0.cfg
│  │  ├─ yolo.cfg
│  │  ├─ yolo9000.cfg
│  │  ├─ yolov1
│  │  │  ├─ tiny-coco.cfg
│  │  │  ├─ tiny-yolo.cfg
│  │  │  ├─ xyolo.test.cfg
│  │  │  ├─ yolo-coco.cfg
│  │  │  ├─ yolo-small.cfg
│  │  │  ├─ yolo.cfg
│  │  │  ├─ yolo.train.cfg
│  │  │  └─ yolo2.cfg
│  │  ├─ yolov2-tiny-voc.cfg
│  │  ├─ yolov2-tiny.cfg
│  │  ├─ yolov2-voc.cfg
│  │  ├─ yolov2.cfg
│  │  ├─ yolov3-openimages.cfg
│  │  ├─ yolov3-spp.cfg
│  │  ├─ yolov3-tiny-prn.cfg
│  │  ├─ yolov3-tiny.cfg
│  │  ├─ yolov3-tiny_3l.cfg
│  │  ├─ yolov3-tiny_obj.cfg
│  │  ├─ yolov3-tiny_occlusion_track.cfg
│  │  ├─ yolov3-tiny_xnor.cfg
│  │  ├─ yolov3-voc.cfg
│  │  ├─ yolov3-voc.yolov3-giou-40.cfg
│  │  ├─ yolov3.cfg
│  │  ├─ yolov3.coco-giou-12.cfg
│  │  ├─ yolov3_5l.cfg
│  │  ├─ yolov4-custom.cfg
│  │  ├─ yolov4-tiny-3l.cfg
│  │  ├─ yolov4-tiny-custom.cfg
│  │  ├─ yolov4-tiny.cfg
│  │  ├─ yolov4-tiny_contrastive.cfg
│  │  └─ yolov4.cfg
│  ├─ cmake
│  │  ├─ Modules
│  │  │  ├─ FindCUDNN.cmake
│  │  │  ├─ FindPThreads_windows.cmake
│  │  │  └─ FindStb.cmake
│  │  ├─ vcpkg_linux.diff
│  │  ├─ vcpkg_linux_cuda.diff
│  │  ├─ vcpkg_osx.diff
│  │  ├─ vcpkg_windows.diff
│  │  └─ vcpkg_windows_cuda.diff
│  ├─ darknet
│  ├─ darknet.py
│  ├─ darknet_images.py
│  ├─ darknet_video.py
│  ├─ image_yolov3.sh
│  ├─ image_yolov4.sh
│  ├─ include
│  │  ├─ darknet.h
│  │  └─ yolo_v2_class.hpp
│  ├─ json_mjpeg_streams.sh
│  ├─ libdarknet.so
│  ├─ net_cam_v3.sh
│  ├─ net_cam_v4.sh
│  ├─ obj
│  │  ├─ activation_kernels.o
│  │  ├─ activation_layer.o
│  │  ├─ activations.o
│  │  ├─ art.o
│  │  ├─ avgpool_layer.o
│  │  ├─ avgpool_layer_kernels.o
│  │  ├─ batchnorm_layer.o
│  │  ├─ blas.o
│  │  ├─ blas_kernels.o
│  │  ├─ box.o
│  │  ├─ captcha.o
│  │  ├─ cifar.o
│  │  ├─ classifier.o
│  │  ├─ coco.o
│  │  ├─ col2im.o
│  │  ├─ col2im_kernels.o
│  │  ├─ compare.o
│  │  ├─ connected_layer.o
│  │  ├─ conv_lstm_layer.o
│  │  ├─ convolutional_kernels.o
│  │  ├─ convolutional_layer.o
│  │  ├─ cost_layer.o
│  │  ├─ crnn_layer.o
│  │  ├─ crop_layer.o
│  │  ├─ crop_layer_kernels.o
│  │  ├─ dark_cuda.o
│  │  ├─ darknet.o
│  │  ├─ data.o
│  │  ├─ demo.o
│  │  ├─ detection_layer.o
│  │  ├─ detector.o
│  │  ├─ dice.o
│  │  ├─ dropout_layer.o
│  │  ├─ dropout_layer_kernels.o
│  │  ├─ gaussian_yolo_layer.o
│  │  ├─ gemm.o
│  │  ├─ go.o
│  │  ├─ gru_layer.o
│  │  ├─ http_stream.o
│  │  ├─ im2col.o
│  │  ├─ im2col_kernels.o
│  │  ├─ image.o
│  │  ├─ image_opencv.o
│  │  ├─ layer.o
│  │  ├─ list.o
│  │  ├─ local_layer.o
│  │  ├─ lstm_layer.o
│  │  ├─ matrix.o
│  │  ├─ maxpool_layer.o
│  │  ├─ maxpool_layer_kernels.o
│  │  ├─ network.o
│  │  ├─ network_kernels.o
│  │  ├─ nightmare.o
│  │  ├─ normalization_layer.o
│  │  ├─ option_list.o
│  │  ├─ parser.o
│  │  ├─ region_layer.o
│  │  ├─ reorg_layer.o
│  │  ├─ reorg_old_layer.o
│  │  ├─ rnn.o
│  │  ├─ rnn_layer.o
│  │  ├─ rnn_vid.o
│  │  ├─ route_layer.o
│  │  ├─ sam_layer.o
│  │  ├─ scale_channels_layer.o
│  │  ├─ shortcut_layer.o
│  │  ├─ softmax_layer.o
│  │  ├─ super.o
│  │  ├─ swag.o
│  │  ├─ tag.o
│  │  ├─ tree.o
│  │  ├─ upsample_layer.o
│  │  ├─ utils.o
│  │  ├─ voxel.o
│  │  ├─ writing.o
│  │  ├─ yolo.o
│  │  └─ yolo_layer.o
│  ├─ results
│  │  └─ comp4_det_test_aircraft.txt
│  ├─ scripts
│  │  ├─ README.md
│  │  ├─ dice_label.sh
│  │  ├─ gen_anchors.py
│  │  ├─ gen_tactic.sh
│  │  ├─ get_coco2017.sh
│  │  ├─ get_coco_dataset.sh
│  │  ├─ get_imagenet_train.sh
│  │  ├─ get_openimages_dataset.py
│  │  ├─ imagenet_label.sh
│  │  ├─ install_OpenCV4.sh
│  │  ├─ kitti2yolo.py
│  │  ├─ kmeansiou.c
│  │  ├─ log_parser
│  │  │  ├─ log_parser.py
│  │  │  ├─ plot.jpg
│  │  │  ├─ readme.md
│  │  │  ├─ run_log_parser_windows.cmd
│  │  │  ├─ test.log
│  │  │  ├─ test_new.log
│  │  │  └─ test_new.svg
│  │  ├─ reval_voc.py
│  │  ├─ reval_voc_py3.py
│  │  ├─ setup.ps1
│  │  ├─ setup.sh
│  │  ├─ testdev2017.txt
│  │  ├─ voc_eval.py
│  │  ├─ voc_eval_py3.py
│  │  ├─ voc_label.py
│  │  ├─ voc_label_difficult.py
│  │  └─ windows
│  │     ├─ otb_get_labels.sh
│  │     ├─ win_cifar.cmd
│  │     ├─ win_get_imagenet_train_48hours.cmd
│  │     ├─ win_get_imagenet_valid.cmd
│  │     ├─ win_get_otb_datasets.cmd
│  │     ├─ win_install_cygwin.cmd
│  │     ├─ windows_imagenet_label.sh
│  │     └─ windows_imagenet_train.sh
│  ├─ src
│  │  ├─ activation_kernels.cu
│  │  ├─ activation_layer.c
│  │  ├─ activation_layer.h
│  │  ├─ activations.c
│  │  ├─ activations.h
│  │  ├─ art.c
│  │  ├─ avgpool_layer.c
│  │  ├─ avgpool_layer.h
│  │  ├─ avgpool_layer_kernels.cu
│  │  ├─ batchnorm_layer.c
│  │  ├─ batchnorm_layer.h
│  │  ├─ blas.c
│  │  ├─ blas.h
│  │  ├─ blas_kernels.cu
│  │  ├─ box.c
│  │  ├─ box.h
│  │  ├─ captcha.c
│  │  ├─ cifar.c
│  │  ├─ classifier.c
│  │  ├─ classifier.h
│  │  ├─ coco.c
│  │  ├─ col2im.c
│  │  ├─ col2im.h
│  │  ├─ col2im_kernels.cu
│  │  ├─ compare.c
│  │  ├─ connected_layer.c
│  │  ├─ connected_layer.h
│  │  ├─ conv_lstm_layer.c
│  │  ├─ conv_lstm_layer.h
│  │  ├─ convolutional_kernels.cu
│  │  ├─ convolutional_layer.c
│  │  ├─ convolutional_layer.h
│  │  ├─ cost_layer.c
│  │  ├─ cost_layer.h
│  │  ├─ cpu_gemm.c
│  │  ├─ crnn_layer.c
│  │  ├─ crnn_layer.h
│  │  ├─ crop_layer.c
│  │  ├─ crop_layer.h
│  │  ├─ crop_layer_kernels.cu
│  │  ├─ dark_cuda.c
│  │  ├─ dark_cuda.h
│  │  ├─ darknet.c
│  │  ├─ darkunistd.h
│  │  ├─ data.c
│  │  ├─ data.h
│  │  ├─ deconvolutional_kernels.cu
│  │  ├─ deconvolutional_layer.c
│  │  ├─ deconvolutional_layer.h
│  │  ├─ demo.c
│  │  ├─ demo.h
│  │  ├─ detection_layer.c
│  │  ├─ detection_layer.h
│  │  ├─ detector.c
│  │  ├─ dice.c
│  │  ├─ dropout_layer.c
│  │  ├─ dropout_layer.h
│  │  ├─ dropout_layer_kernels.cu
│  │  ├─ gaussian_yolo_layer.c
│  │  ├─ gaussian_yolo_layer.h
│  │  ├─ gemm.c
│  │  ├─ gemm.h
│  │  ├─ getopt.c
│  │  ├─ getopt.h
│  │  ├─ gettimeofday.c
│  │  ├─ gettimeofday.h
│  │  ├─ go.c
│  │  ├─ gru_layer.c
│  │  ├─ gru_layer.h
│  │  ├─ http_stream.cpp
│  │  ├─ http_stream.h
│  │  ├─ httplib.h
│  │  ├─ im2col.c
│  │  ├─ im2col.h
│  │  ├─ im2col_kernels.cu
│  │  ├─ image.c
│  │  ├─ image.h
│  │  ├─ image_opencv.cpp
│  │  ├─ image_opencv.h
│  │  ├─ layer.c
│  │  ├─ layer.h
│  │  ├─ list.c
│  │  ├─ list.h
│  │  ├─ local_layer.c
│  │  ├─ local_layer.h
│  │  ├─ lstm_layer.c
│  │  ├─ lstm_layer.h
│  │  ├─ matrix.c
│  │  ├─ matrix.h
│  │  ├─ maxpool_layer.c
│  │  ├─ maxpool_layer.h
│  │  ├─ maxpool_layer_kernels.cu
│  │  ├─ network.c
│  │  ├─ network.h
│  │  ├─ network_kernels.cu
│  │  ├─ nightmare.c
│  │  ├─ normalization_layer.c
│  │  ├─ normalization_layer.h
│  │  ├─ option_list.c
│  │  ├─ option_list.h
│  │  ├─ parser.c
│  │  ├─ parser.h
│  │  ├─ region_layer.c
│  │  ├─ region_layer.h
│  │  ├─ reorg_layer.c
│  │  ├─ reorg_layer.h
│  │  ├─ reorg_old_layer.c
│  │  ├─ reorg_old_layer.h
│  │  ├─ rnn.c
│  │  ├─ rnn_layer.c
│  │  ├─ rnn_layer.h
│  │  ├─ rnn_vid.c
│  │  ├─ route_layer.c
│  │  ├─ route_layer.h
│  │  ├─ sam_layer.c
│  │  ├─ sam_layer.h
│  │  ├─ scale_channels_layer.c
│  │  ├─ scale_channels_layer.h
│  │  ├─ shortcut_layer.c
│  │  ├─ shortcut_layer.h
│  │  ├─ softmax_layer.c
│  │  ├─ softmax_layer.h
│  │  ├─ super.c
│  │  ├─ swag.c
│  │  ├─ tag.c
│  │  ├─ tree.c
│  │  ├─ tree.h
│  │  ├─ upsample_layer.c
│  │  ├─ upsample_layer.h
│  │  ├─ utils.c
│  │  ├─ utils.h
│  │  ├─ version.h
│  │  ├─ version.h.in
│  │  ├─ voxel.c
│  │  ├─ writing.c
│  │  ├─ yolo.c
│  │  ├─ yolo_console_dll.cpp
│  │  ├─ yolo_layer.c
│  │  ├─ yolo_layer.h
│  │  └─ yolo_v2_class.cpp
│  ├─ uselib
│  ├─ video_yolov3.sh
│  └─ video_yolov4.sh
└─ run.py

```