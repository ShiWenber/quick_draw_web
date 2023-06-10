
import glob
import io
import numpy as np
import torch
import os
# import cv2
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
from fu_cnn import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(10).to(device)
model.load_state_dict(torch.load('CNN_model.pth'))
model.eval()
class_dict = {0: 'ambulance', 1: 'apple', 2: 'bear', 3: 'bicycle', 4: 'bird', 5: 'bus', 6: 'cat', 7: 'foot', 8: 'owl', 9: 'pig'}

def inference(img_data: np.array, path: str = None, uuid_str: str = None):
    """ 图像分类

    Args:
        img (np.ndarray): 输入图像
        path (str, optional): 从本地路径图片. Defaults to None.
        uuid_str (str, optional): 处理任务uuid编号. Defaults to None.

    Returns:
        string: 分类结果字符串
    """
    # 生成uuid标识任务
    if uuid_str == None:
        uuid_str = uuid.uuid4().hex
    # 如果path 不为none，使用path
    if path != None:
        print("path:", path)
        img_data = None
        if path.endswith(".npy"):
            # 读取
            img_data = np.load(path)
        else:
            # 以灰度图像读取
            img = Image.open(path).convert('L')
            # resize 为 28 * 28，并用抗锯齿算法
            img = img.resize((28, 28), Image.ANTIALIAS)
            img_data = np.array(img)

    start_time = time.time()

    # print("img_data:", img_data)
    
    img_data = img_data.reshape(-1, 28, 28)

    img = torch.from_numpy(img_data)
    img = img.float() / 255.0
    img = img.view(-1, 1, 28, 28).to(device)
    outputs = model(img)
    # print("outputs:", outputs)
    _, predicted = torch.max(outputs, 1)

    outputs = torch.nn.functional.softmax(outputs, dim=1)
    # 获得概率分布 dict
    predict_class = {}
    class_num = len(class_dict)
    for i in range(class_num):
        predict_class[class_dict[i]] = outputs[0][i].item()
        
    print("predict_class:", predict_class)

    # assert predicted.shape[0] == 1
    # print("predicted:", predicted)
    # print("predicted.max:", predicted.max())

    print(f"{uuid_str} time:", time.time() - start_time)
    print('predicted: ', predicted)
    return class_dict[predicted.item()], predict_class

# ----test for inference
# print(inference(np.load("data/quick_draw_data/quick_draw_data.npy")))
# image = Image.open("data/quick_draw_data/quick_draw_data.jpg")
# image_data = np.array(image)
# print(image_data)
# print(inference(image_data))
# exit(0)

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
    return "backend is running"

@app.route("/getResTest",methods = ["POST"])
@swag_from("get_img_res.yml")
def get_img():
    uuid_str = uuid.uuid4().hex
    # 获取上传文件的类型 FileStorage
    file = request.files['file']
    # 文件后缀
    base_path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(base_path, "temp")):
        os.makedirs(os.path.join(base_path, "temp"))
    file_name = secure_filename(uuid_str + file.filename)
    upload_path = os.path.join(base_path, "temp", file_name)
    file.save(upload_path)
    res = {"class_name": "Not recognized", "predict_class": {}}
    try:
        class_name, predict_class = inference(None, path=upload_path)
        res = {"class_name": class_name, "predict_class": predict_class}
    except Exception as e:
        print(e)
        pass
    # 删除临时文件
    os.remove(upload_path)

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
    # image_array = np.frombuffer(image_binary_data, np.int)

    # # 转化为opencv图像的ndarray格式
    # image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    # 读取为PIL图像
    image = Image.open(BytesIO(image_binary_data))

    image_data = np.array(image)

    # 断言检查image的类型是否为ndarray
    # assert type(image_array) is np.array

    # 图像处理部分
    res = {"class_name": "Not recognized", "predict_class": {}}
    try:
        class_name, predict_class = inference(image_data)
        res = {"class_name": class_name, "predict_class": predict_class}
    except Exception as e:
        print(e)
        pass

    return jsonify(res)

# 查看每个类的示例图片
@app.route("/getExample",methods = ["POST"])
@swag_from("get_example.yml")
def get_example():
    data = request.get_json()
    class_name_2_num_dict = data['class_name_2_num_dict']
    assert type(class_name_2_num_dict) is dict
    # 判断 class_name_2_num_dict 是否所有包含全局定义的类别
    assert len(class_name_2_num_dict) == len(class_dict)
    for id, class_name in class_dict.items():
        if class_name not in class_name_2_num_dict:
            class_name_2_num_dict[class_name] = 0
    print("class_name_2_num_dict:", class_name_2_num_dict)

    data_dir='data/quick_draw_data'
    listes=os.listdir(data_dir)
    print(listes)
    data={}

    for i in range(len(listes)):
        data[listes[i]]=(np.load(data_dir+'/'+listes[i]+'/'+listes[i]+'.npy'))


    random_array = np.random.randint(0, data[listes[0]].shape[0], size=20)

    res = {}
    for i in range(len(listes)):
        res[listes[i]] = []
    
    # print("共有十类，每类随机取得十张图片如下：")
    # for j in range(10):
    for j in range(len(listes)):
        for i in range(class_name_2_num_dict[listes[j]]):
            image_npy = data[listes[j]][random_array[i]].reshape(28, 28).astype(np.uint8)
            # 将内存中的图像转换为base64字符串
            img_base64_str = imageToBase64(image_npy)
            res[listes[j]].append(img_base64_str)
    # print(res)

    return jsonify(res)


# @app.route("/getRes",methods = ["POST"])
# @swag_from("get_res.yml")
# def res():
#     data = request.get_json()
#     url = data['url']
#     image = OssUtils.read_img_by_url(url)

#     # 断言检查image的类型是否为ndarray
#     assert type(image) is np.ndarray

#         # 图像处理部分
#     res = {"disease_en": "Not recognized", "disease_zh": "未识别", "disease_id": ''}
#     try:
#         disease_en, disease_zh, disease_id = pred.predict_img(image)
#         res = {"disease_en": disease_en, "disease_zh": disease_zh, "disease_id": disease_id}
#     except Exception as e:
#         print(e)
#         pass


#     # 将内存中的图像上传到oss TODO


#     return jsonify(res)

# 将内存中的图像转换为 Base64 字符串
def imageToBase64(image_npy):
    # 将 NumPy 数组转换为 PIL 图像
    image = Image.fromarray(image_npy)

    # 创建一个内存文件对象
    buffer = io.BytesIO()

    # 将 PIL 图像保存为 JPEG 格式到内存文件对象中
    image.save(buffer, format='JPEG')

    # 从内存文件对象中获取字节串
    image_data = buffer.getvalue()

    # 将字节串编码为 Base64 字符串
    img_base64_str = base64.b64encode(image_data).decode('utf-8')

    # 返回 Base64 字符串
    return img_base64_str


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5003,debug=True)
