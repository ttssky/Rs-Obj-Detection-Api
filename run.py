from app.main import multiThreadRequest
from app import create_app
from config import *

app = create_app('default')
app.run(host='0.0.0.0', port=5000, debug=True)

# if __name__ == "__main__":
#     total_bbox = [14719942.5335246566683054,4046216.8749611256644130, 
#                     14722287.1716489437967539,4050584.9443750921636820]
#     width_min = total_bbox[0]
#     height_min = total_bbox[1]
#     width_max = total_bbox[2]
#     height_max = total_bbox[3]

#     image_id = "37605aef-913e-40b6-8859-822e72a51a19"
#     params = {
#     "bbox": [],
#     "bands": [0,1,2],
#     "height": 608,
#     "width": 608
#     }
#     headers = {
#     "x-heycloud-admin-session": "tFn8aWIgWHnYTCO/NR/r2OK4wef96gtC",
#     "Content-Type":"application/json"
# }


#     request_url = "http://192.168.31.17:9001/heycloud/api/data/idataset/37605aef-913e-40b6-8859-822e72a51a19/extract"
#     image_height = 12715
#     image_width = 6825
#     threads = multiThreadRequest(INFER_IMAGE_PATH,
#                         height_min,
#                         width_min,
#                         height_max,
#                         width_max,
#                         SLICE_HEIGHT,
#                         SLICE_WIDTH,
#                         SP_SLICE_HEIGHT,
#                         SP_SLICE_WIDTH,
#                         image_height,
#                         image_width,
#                         SLICE_OVERLAP,
#                         params,
#                         request_url,
#                         headers
#                         )
#     print(len(threads))
#     for th in threads:
#         th.setDaemon(True)
#         th.start()
#     for t in threads:
#         t.join()
