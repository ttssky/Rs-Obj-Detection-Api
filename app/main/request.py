from requests import post
from threading import Thread
import json
import numpy as np
from PIL import Image
import time
import os
from glob import glob

def _extract_windows(ims_dir,
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
                    overlap):
    
    x0, y0 = width_min, height_min
    dx = float((sp_slice_width - overlap * sp_slice_width))
    dy = float((sp_slice_height - overlap * sp_slice_height))
    n_ims = 0
    windows = []
    save_path = []
    while y0 < height_max :
        while x0 < width_max:

            n_ims += 1
            if (n_ims % 50 == 0):
                print(n_ims)
            if y0 + sp_slice_height > height_max:
                y = height_max - sp_slice_height
            else:
                y = y0
            if x0 + sp_slice_width > width_max:
                x = width_max - sp_slice_width
            else:
                x = x0
            window_extract = [x, y, x + sp_slice_width, y + sp_slice_height]

           
            outpath = os.path.join(
                ims_dir,
                'test' + '__' + str(y) + '_' + str(x) + '_'
                + str(slice_height) + '_' + str(slice_width)
                + '_' + '0' + '_' + str(image_width) + '_' + str(image_height)
                + '.png')
            save_path.append(outpath)
            windows.append(window_extract)
            x0 = x0 + dx

        x0 = width_min
        y0 = y0 + dy

    return windows, save_path

def _bytes_to_images(response, save_path, *args, **kwargs):
    
    if response.status_code == 200:
        arr = np.frombuffer(response.content, dtype=np.uint8)
        size = int(arr.size / 3)
        b0 = arr[0:size]
        b1 = arr[size:2*size]
        b2 = arr[2*size:3*size]

        img0 = Image.frombytes(mode="L",size=(608,608),data=b0,decoder_name="raw")
        img1 = Image.frombytes(mode="L",size=(608,608),data=b1,decoder_name="raw")
        img2 = Image.frombytes(mode="L",size=(608,608),data=b2,decoder_name="raw")

        Image.merge("RGB", (img0, img1, img2)).save(save_path)
    else:
        print(response.content)

def _async_request(request_url, params, headers, callback=None, save_path='', **kwargs):
    if callback:
        def callback_with_args(response, *args, **kwargs):
            print(save_path)
            kwargs['save_path'] = save_path
            callback(response, *args, **kwargs)
        kwargs['hooks'] = {'response': callback_with_args}

    kwargs['data'] = params
    kwargs['headers'] = headers
    kwargs['url'] = request_url
    kwargs['timeout'] = 60
    thread = Thread(target=post, kwargs=kwargs)
    return thread
    # thread.start()
    # thread.join()

def multiThreadRequest(ims_dir,
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
                       overlap,
                       post_params,
                       request_url,
                       post_headers):

    windows, save_path = _extract_windows(ims_dir,
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
                                          overlap)

    threads = []
    for window, path in zip(windows, save_path):
        post_params['bbox'] = window
        try:
            threads.append(_async_request(request_url,
                                          json.dumps(post_params),
                                          post_headers,
                                          callback=_bytes_to_images,
                                          save_path=path))
        except:
            continue

    return threads
