
import glob
import numpy as np
import torch
import os
import cv2
import time
from inference import predict_class

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import uuid
from PIL import Image
import os
import time
import base64
import json
from flasgger import Swagger, swag_from
from io import BytesIO
from oss_utils import OssUtils

pred = predict_class()
# img_path="/root/models/作物病害/val/Apple___Apple_scab/0d3c0790-7833-470b-ac6e-94d0a3bf3e7c___FREC_Scab 2959.JPG"
# print(pred.predict(img_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)
Swagger(app)
# app.config['JSON_AS_ASCII'] = False # 解决中文乱码问题

@app.route("/test",methods = ["GET"])
def test():
    # 用于测试服务是否并行
    time.sleep(1)
    return "0"


@app.route("/",methods=["GET"])
def show():
    return "leaf disease classification api"

@app.route("/getResTest",methods = ["POST"])
@swag_from("get_img_res.yml")
def get_img():
    file = request.files['file']
    base_path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(base_path, "temp")):
        os.makedirs(os.path.join(base_path, "temp"))
    file_name = uuid.uuid4().hex
    upload_path = os.path.join(base_path, "temp", file_name)
    file.save(upload_path)
    res = pred.predict(upload_path)
    # 删除临时文件
    os.remove(upload_path)
    res = {"status": res}
    # 返回图片
    ## 全部在内存缓冲区完成能提高性能（如果使用imwrite再保存会导致从外部磁盘读取的io操作）
    return jsonify(res)

@app.route("/getBase64Res",methods = ["POST"])
@swag_from("get_base64_res.yml")
def image():
    data = request.get_json()
    image_data = data['image_data']

    # 将base64字符串转换为二进制数据
    image_binary_data = base64.b64decode(image_data)

    # 转换二进制数据为numpy数组
    image_array = np.frombuffer(image_binary_data, np.uint8)

    # 转化为opencv图像的ndarray格式
    image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    # 断言检查image的类型是否为ndarray
    assert type(image) is np.ndarray

    # 图像处理部分
    # res_img = segmentation(image)
    res = pred.predict_img(image)

    return jsonify({'status': res})


@app.route("/getRes",methods = ["POST"])
@swag_from("get_res.yml")
def res():
    data = request.get_json()
    url = data['url']
    image = OssUtils.read_img_by_url(url)

    # 断言检查image的类型是否为ndarray
    assert type(image) is np.ndarray

    # 图像处理部分
    res = pred.predict_img(image)

    # 将内存中的图像上传到oss TODO

    return jsonify({'status': res})


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5005,debug=True)
