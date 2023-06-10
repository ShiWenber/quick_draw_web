#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/5/19 19:16
# @Author  : fuchanglong
# @File    : CNN.py
import argparse
import os

import numpy as np
import torch
# from tqdm import tqdm
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
from torch import nn
# plt.rcParams['font.family']='DejaVu Sans'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_data():
    # 读取数据
    path = './data/quick_draw_data/'
    path_list = os.listdir(path)
    random_data = []
    random_y = []
    # 生成标签字典
    map_labels = {i: path_list[i] for i in range(len(path_list))}
    print(map_labels)
    # 初始化数据
    X_data = np.empty((0, 28, 28))
    y_data = np.empty((0,))
    # 读取数据
    for index, i in enumerate(path_list):
        full_path = os.path.join(path, i, i + '.npy')
        data = np.load(full_path)
        # 选取10个样本
        images = data.reshape(-1, 28, 28)
        sample_indices = np.random.choice(data.shape[0], size=10, replace=False)
        random_data.append(images[sample_indices].tolist())
        my_labels = np.full((20000,), index, dtype=np.int64)
        random_y.append(my_labels[sample_indices].tolist())
        # 将数据添加到总数据中
        X_data = np.concatenate((X_data, images), axis=0)
        y_data = np.append(y_data, my_labels)
    random_data = np.array(random_data)
    random_y = np.array(random_y)
    return X_data, y_data, map_labels, random_data, random_y


def create_data_loader(data, labels, batch_size, num_workers=4):
    # 创建数据加载器
    data_tensor = torch.from_numpy(data).float()
    label_tensor = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)
    # 创建数据加载器
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader

# 定义超参数
class CNN(nn.Module):
    def __init__(self, num_classes):
        # 初始化父类
        super(CNN, self).__init__()
        #  定义卷积层
        self.num_classes = num_classes
        # 定义卷积层，卷积核大小为3，步长为1，padding为1
        # 池化层，池化核大小为2，步长为2
        # 两个卷积层，两个池化层
        # 定义全连接层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 定义全连接层
        self.fc_layer = nn.Sequential(
            # 将图像形状变为 (batch_size, 32 * 7 * 7)
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            # 使用ReLU激活函数
            nn.ReLU(),
            # 使用Dropout
            nn.Linear(128, self.num_classes)
        )

    def forward(self, x):
        # 前向传播
        x = self.conv_layers(x)
        # 将卷积层的输出拉直
        x = self.fc_layer(x)
        return x


def train(args, model, train_loader, val_loader):
    # 定义模型

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    losses=[]
    acc=[]
    bp = tqdm(range(num_epochs))
    for epoch in bp:
        ls=0
        for images, labels in train_loader:
            # 将数据转换为张量，并进行标准化处理
            images = images.float() / 255.0
            images = images.view(-1, 1, 28, 28).to(device)  # 将图像形状变为 (batch_size, 1, 28, 28)
            labels = labels.long().to(device)

            # 前向传播和计算损失
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ls+=loss.item()
        losses.append(ls/len(bp))
        # 输出每个epoch的损失和准确率
        with torch.no_grad():
            total_loss = 0.0
            total_correct = 0

            for images, labels in val_loader:
                images = images.float() / 255.0
                images = images.view(-1, 1, 28, 28).to(device)  # 将图像形状变为 (batch_size, 1, 28, 28)
                labels = labels.long().to(device)
                # 前向传播和计算损失
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
            # 输出每个epoch的损失和准确率
            bp.set_description('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, total_loss / len(test_loader),total_correct / len(test_loader.dataset) * 100))
            acc.append(total_correct / len(test_loader.dataset))
            # 保存模型
    torch.save(model.state_dict(),'CNN_model.pth')
    return  model, losses,acc


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default='0', help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to train (default: 100)')
    parser.add_argument('--num_classes', type=int, default=10, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    args = parser.parse_known_args()[0]
    print(args)
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.num_epochs
    lr = args.lr
    torch.manual_seed(0)
    np.random.seed(0)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    # 读取数据
    X_data, y_data, map_labels, random_data, random_y = read_data()
    # 划分数据集
    train_data, val_test_data, train_y, val_test_y = train_test_split(X_data, y_data, test_size=0.4, random_state=42)
    val_data, test_data, val_y, test_y = train_test_split(val_test_data, val_test_y, test_size=0.5, random_state=42)
    # 创建数据加载器
    train_loader = create_data_loader(train_data, train_y, batch_size=batch_size, num_workers=num_workers)
    val_loader = create_data_loader(val_data, val_y, batch_size=batch_size, num_workers=num_workers)
    test_loader = create_data_loader(test_data, test_y, batch_size=batch_size, num_workers=num_workers)
    # 创建模型
    num_classes = args.num_classes
    model = CNN(num_classes=10).to(device)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model,losses,acc= train(args, model, train_loader, val_loader)
    # 绘制损失和准确率曲线
    plt.plot(range(1,len(losses)+1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over time')
    plt.show()
    plt.plot(range(1,len(acc)+1), acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over time')
    plt.show()
    # 统计结果
    # 绘制预测结果，每行5个，一共10行
    fig, ax = plt.subplots(10, 5, figsize=(12, 20))
    accuracy_dict = {}
    for index, (x, y) in enumerate(zip(random_data, random_y)):
        loader = create_data_loader(x, y, batch_size=batch_size, num_workers=num_workers)
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            # 遍历数据集
            for images, labels in loader:
                images = images.float() / 255.0
                images = images.view(-1, 1, 28, 28).to(device)
                labels = labels.long().to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                # 在子图的右侧添加预测标签
                for j, ig in enumerate(images):
                    if j >= 5:
                        break
                    # 将预测标签转换为对应的文字
                    predicted_label = map_labels[predicted[j].item()]
                    ax[index][j].imshow(ig.squeeze(0).cpu().numpy())
                    ax[index][j].set_xticks([])
                    ax[index][j].set_yticks([])
                    color = 'black' if labels[j] == predicted[j] else 'red'
                    # 设置标题颜色
                    ax[index][j].set_title(f"Predicted: {predicted_label}", color=color)
                title = ax[index][0].set_ylabel(f"{map_labels[index]}", fontsize=14, color='blue')
            # 计算准确率
            accuracy = total_correct / total_samples
            accuracy_dict[map_labels[index]] = accuracy
            print(f"Class of {map_labels[index]} Accuracy: {total_correct / 10 * 100}")

    # 创建柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(accuracy_dict)), list(accuracy_dict.values()), align='center', width=0.8)
    plt.xticks(range(len(accuracy_dict)), list(accuracy_dict.keys()))
    plt.ylabel('Accuracy')
    plt.title('Accuracy of each category')

    # 在每个柱状图上方添加准确率的值
    for i, v in enumerate(list(accuracy_dict.values())):
        plt.text(i, v + 0.01, f"{v:.1%}", ha='center', fontweight='bold')

    plt.show()
