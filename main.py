
import glob
import io
import numpy as np
import torch
import os
import cv2
import time

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
# from fu_lstm import BiLSTM
from LSTM import BiLSTM

from rdp import rdp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(10).to(device)
model.load_state_dict(torch.load('CNN_model.pth'))
model.eval()

model2 = BiLSTM(5, 256, 1, 10).to(device)
model2.load_state_dict(torch.load('LSTM_model_fu.pth'))
model2.eval()

class_dict = {0: 'ambulance', 1: 'apple', 2: 'bear', 3: 'bicycle', 4: 'bird', 5: 'bus', 6: 'cat', 7: 'foot', 8: 'owl', 9: 'pig'}

import numpy as np


def convert_txt_to_npy_no_fit(txt_filename, npy_filename):
    sequences = []
    with open(txt_filename, 'r') as file:
        for line in file:
            line = line.strip()
            x, y, p = map(float, line.split(','))  # 假设文本文件中的数据以逗号分隔
            sequences.append([x, y, p])
    npy_sequences = np.array(sequences)

    points = []
    with open("data.txt", 'r') as f:
        for line in f:
            x, y, p = line.split(',')
            points.append([int(x),int(y), int(p)])
    npy_data = np.array(points)
    print(npy_data.shape)
    npy_data_5 = np.zeros((len(npy_data), 5))

    # 转化为偏移量数组    
    x0 = npy_data[0][0]
    y0 = npy_data[0][1]
    dx = 0
    dy = 0

    for i in range(len(npy_data)):
        x, y, p = npy_data[i]
        dx = x - x0
        dy = y - y0
        p1 = int(p == 0)
        p2 = int(p == 1)
        # p3 = int((i + 1) % 151 == 0)
        p3 = 0
        npy_data_5[i] = np.array([dx, dy, p1, p2, p3])
        x0 = x
        y0 = y
    # 删除第一行
    npy_data_5 = npy_data_5[1:]
    npy_data_5[-1][-1] = 1
    # print(npy_data_5)
    # 保证最后一个点的最后一个维度为1
    # 转换为numpy数组并保存为.npy文件
    np.save(npy_filename, npy_data_5)
    return npy_data_5

    

def convert_txt_to_npy(txt_filename, npy_filename, epsilon=1.2, data_len=151):
    sequences = []
    with open(txt_filename, 'r') as file:
        for line in file:
            line = line.strip()
            x, y, p = map(float, line.split(','))  # 假设文本文件中的数据以逗号分隔
            sequences.append([x, y, p])
    npy_sequences = np.array(sequences)
    # points = np.zeros((len(column3), 2))
    
    # # 可视化
    # x = 0
    # y = 0
    # for dx, dy, _ in npy_filename:
    #     # 起笔
    #     if p1 == 1:
    #         plt.plot([x, x + dx], [y, y + dy], color='black')
    #     # 提笔 
    #     if p3 == 1:
    #         break
    #     x += dx
    #     y += dy    

    sequences = rdp(npy_sequences, epsilon=epsilon)

    # 做一个反馈处理，只有当 sequences 的长度大于 151 时，才继续
    max_iter = 20 # 调整的最大次数
    iter = 0
    while( data_len - 75 >= len(sequences) or len(sequences) > data_len):
        iter += 1
        if len(sequences) > data_len:
            epsilon += 0.1
        else:
            epsilon -= 0.1
        sequences = rdp(npy_sequences, epsilon=epsilon)
        if iter > max_iter:
            break

    print('epsilon: ', epsilon)
    print('sequences len: ', len(sequences))
    x0 = sequences[0][0]
    y0 = sequences[0][1]
    dx = 0
    dy = 0

    # 将三维数据转换为五维数据
    converted_sequences = []
    for i in range(len(sequences)):
        x, y, p = sequences[i]
        dx = x - x0
        dy = y - y0
        p1 = int(p == 0)
        p2 = int(p == 1)
        p3 = int((i + 1) % 151 == 0)
        converted_sequences.append([dx, dy, p1, p2, p3])
        x0 = x
        y0 = y
        # 截取前151个点
        if i >= data_len-1:
            break
    converted_sequences = converted_sequences[1:]
    converted_sequences[-1][-1] = 1
    print(converted_sequences)
    # 保证最后一个点的最后一个维度为1
    converted_sequences[-1][-1] = 1
    # 转换为numpy数组并保存为.npy文件
    converted_sequences = np.array(converted_sequences)
    # np.save(npy_filename, converted_sequences)
    return converted_sequences

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
            # Image.fromarray(np.array(img)).save(f"test_gray.png")
            img_data = np.array(img)
            img_data[img_data > 0] = 255
            img = Image.fromarray(img_data)
            # Image.fromarray(img_data).save(f"test_gray2.png")
            # resize 为 28 * 28，并用抗锯齿算法
            # img = img.resize((28, 28), Image.ANTIALIAS)
            img = img.resize((28, 28))
            img_data = np.array(img)
            # print("img_data:", img_data)
            # Image.fromarray(img_data).save(f"test.png")
            # 阈值离散
            img_data[img_data > 0] = 255
            Image.fromarray(img_data).save(f"input.png")
            # 反转图像并标准化处理
            # img_data = np.logical_not(img_data.astype(bool)).astype(int)
            # 保存图像
            # img_pil = Image.fromarray(img_data)
            # img_pil.save(f"{uuid_str}.jpg")

            # print(img_data)

    start_time = time.time()

    # print("img_data:", img_data)
    
    img_data = img_data.reshape(-1, 28, 28)

    img = torch.from_numpy(img_data)
    # 转化为float类型
    img = img.float()
    if img_data.max() > 1:
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

def inference2(path: str=None):

    data_read = None
    if path.endswith(".npy"):
        data_read = np.load(path)
    else:
        data_read = convert_txt_to_npy(path, path)
        # data_read = np.load(path)
    print("data_read.shape:", data_read.shape)

    # 判断data是否为 151, 5
    # 不足151的补 [0,0,0,0,1]
    data = np.zeros((151, 5))
    if data_read.shape[0] < 151:
        for i in range(data_read.shape[0]):
            data[i] = data_read[i]
        for i in range(data_read.shape[0], 151):
            data[i] = [0,0,0,0,1]
    # 超过151的截断
    else:
        for i in range(151):
            data[i] = data_read[i]
        
    assert data.shape[0] <= 151

    tensor_data = torch.from_numpy(data).to(device) # tensor_data.shape = 56, 5
    print("tensor_data.shape:", tensor_data.shape)
    # 将tensor_data转化为float类型
    tensor_data = tensor_data.float()
    print("tensor_data:", tensor_data)
    # 将tensor_data的维度转换为 1, 56, 5
    tensor_data = tensor_data.view(1, tensor_data.shape[0], tensor_data.shape[1])
    # tensor_data = tensor_data.view(-1, 1, tensor_data.shape[0], tensor_data.shape[1])
    # print("tensor_data", tensor_data)
    
    outputs = model2(tensor_data)

    # softmax
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    predict_class = {}
    for i in range(10):
        predict_class[class_dict[i]] = outputs[0][i].item()
    _,predicted = torch.max(outputs, 1)
    print("predict_class:", predict_class)
    print("predict result:", class_dict[predicted[0].item()])
    return class_dict[predicted[0].item()], predict_class

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
    # os.remove(upload_path)

    # 返回图片
    ## 全部在内存缓冲区完成能提高性能（如果使用imwrite再保存会导致从外部磁盘读取的io操作）
    return jsonify(res)

@app.route("/getResTestLstm",methods = ["POST"])
@swag_from("get_img_res_lstm.yml")
def get_img_lstm():
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
        class_name, predict_class = inference2(path=upload_path)
        res = {"class_name": class_name, "predict_class": predict_class}
    except Exception as e:
        print(e)
        pass
    # 删除临时文件
    # os.remove(upload_path)

    # 返回图片
    ## 全部在内存缓冲区完成能提高性能（如果使用imwrite再保存会导致从外部磁盘读取的io操作）
    return jsonify(res)

@app.route("/getBase64ResLstm",methods = ["POST"])
@swag_from("get_base64_res_lstm.yml")
def image_lstm():
    data = request.get_json()
    image_data = data['txt_data']

    # 将base64字符串转换为二进制数据
    npy_file = base64.b64decode(image_data)

    # 保存为npy文件
    # data_read = np.load(BytesIO(npy_file))
    temp_path = "temp/" + uuid.uuid4().hex + ".txt"
    # np.save(temp_path, data_read)
    # 保存为txt文件
    file = open(temp_path, 'wb')
    file.write(npy_file)
    file.close()


    # 转换二进制数据为numpy数组
    # image_array = np.frombuffer(image_binary_data, np.int)

    # # 转化为opencv图像的ndarray格式
    # image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

    # 读取为PIL图像
    # image = Image.open(BytesIO(image_binary_data))

    # 断言检查image的类型是否为ndarray
    # assert type(image_array) is np.array

    # 图像处理部分
    res = {"class_name": "Not recognized", "predict_class": {}}
    try:
        class_name, predict_class = inference2(temp_path)
        res = {"class_name": class_name, "predict_class": predict_class}
    except Exception as e:
        print(e)
        pass

    # 删除临时文件
    # os.remove(temp_path)

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

    # 读取为PIL图像，并且为灰度
    # image = Image.open(BytesIO(image_binary_data))
    image = Image.open(BytesIO(image_binary_data)).convert('L')

    # 调整为28 * 28
    image = image.resize((28, 28), Image.ANTIALIAS)


    image_data = np.array(image)


    # 离散处理
    # image_data[image_data == 255] = 0

    # 保存图像
    Image.fromarray(image_data).save("temp" + ".png")

    # 如果 60% 以上的像素为白色，则认为是白色背景，需要反转
    if image_data[image_data > 220].shape[0] > 0.6 * image_data.shape[0] * image_data.shape[1]:
        image_data = 255 - image_data
    
    Image.fromarray(image_data).save("temp_con" + ".png")


    image_data[image_data > 0] = 255

    Image.fromarray(image_data).save("temp_con2" + ".png")

    # 断言检查image的类型是否为ndarray
    # assert type(image_array) is np.array

    # 图像反转
    # image_data = np.logical_not(image_data.astype(bool)).astype(int)

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
    img_width = data['img_width']
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
            # 将图像放大或者缩小为指定的宽度
            image_npy = cv2.resize(image_npy, (img_width, img_width), interpolation=cv2.INTER_CUBIC)
            # 锐化
            kernel_sharpen_1 = np.array([
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]])
            image_npy = cv2.filter2D(image_npy, -1, kernel_sharpen_1)
            # 阈值离散让图像更加清晰
            # image_npy[image_npy > 125] = 255
            # image_npy[image_npy <= 125] = 0
            # 去锯齿
            # image_npy = cv2.GaussianBlur(image_npy, (3, 3), 0)
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
    # inference2("./data/bus.txt")
    # inference2("./apple_test.npy")
