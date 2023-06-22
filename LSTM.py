import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
plt.rcParams['font.family']='DejaVu Sans'

device = torch.device('cuda:0')


def drawing(data, ax):
    colors = [ 'orange', 'purple',  'gray', 'magenta']
    # 遍历每个点，根据画笔状态绘制笔画轨迹
    x, y = 0, 0  # 初始点坐标
    x_value = [0]
    y_value = [0]
    current = 0
    ax.margins(x=0.1, y=0.1)
    for point in data:
        dx, dy, p1, p2, p3 = point
        if p3 == 1:  # 结束绘画，重置起点
            x, y = 0, 0
            current = 0
            x_value.append(0)
            y_value.append(0)
        elif p2 == 1:  # 离开画布，断开连线
            ax.plot([x, x + dx], [y, y + dy], '-', color=colors[current], linewidth=3)
            x, y = x + dx, y + dy
            x_value.append(x)
            y_value.append(y)
            current = (current + 1) % len(colors)
        elif p1 == 1:  # 正在画图，连线
            ax.plot([x, x + dx], [y, y + dy], '-', color=colors[current], linewidth=3)
            x, y = x + dx, y + dy
            x_value.append(x)
            y_value.append(y)
    # 计算x轴和y轴上的最小值和最大值
    xmin = min(0, min(x_value))
    xmax = max(0, max(x_value))
    ymin = min(0, min(y_value))
    ymax = max(0, max(y_value))  # 设置坐标轴范围
    ax.set_xlim([xmax, xmin])
    ax.set_ylim([ymax,ymin])

    return ax


def read_data():
    path = '../data/sketch_datas/'
    path_list = os.listdir(path)
    random_data = []
    random_y = []
    map_labels = {0: 'ambulance', 1: 'apple', 2: 'bear', 3: 'bicycle', 4: 'bird', 5: 'bus', 6: 'cat', 7: 'foot',
                  8: 'owl', 9: 'pig'}
    print(map_labels)
    y_data = np.empty((0,))
    max_length = 0
    pad_value = [0, 0, 0, 0, 1]  # 填充值
    # 遍历每个文件，读取数据，计算最大长度
    for index, i in enumerate(path_list):
        full_path = os.path.join(path, i)
        data = np.load(full_path)
        if data.shape[1] > max_length:
            max_length = data.shape[1]
    # 根据最大长度对数据进行填充
    X_data = np.empty((0, max_length, 5))
    for index, i in enumerate(path_list):
        full_path = os.path.join(path, i)
        data = np.load(full_path)
        original_length = data.shape[1]
        if data.shape[1] < max_length:
            # 如果数据长度小于最大长度，则在第二维上使用填充值填充
            data = np.pad(data, ((0, 0), (0, max_length - data.shape[1]), (0, 0)), 'constant', constant_values=0)
            data[:, -(max_length - original_length):, :] = pad_value  # 在最后一维上使用填充值填充
        # 将数据添加到X_data中
        X_data = np.concatenate((X_data, data), axis=0)
        my_labels = np.full((70000,), index, dtype=np.int64)
        y_data = np.append(y_data, my_labels)
        sample_indices = np.random.choice(data.shape[0], size=10, replace=False)
        random_data.append(data[sample_indices].tolist())
        random_y.append(my_labels[sample_indices].tolist())
    random_data = np.array(random_data)
    random_y = np.array(random_y)
    return X_data, y_data, map_labels, random_data, random_y


def create_data_loader(data, labels, batch_size, num_workers=4):
    data_tensor = torch.from_numpy(data).float()
    label_tensor = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader


import torch.nn as nn


class BiLSTM(nn.Module):
    # 定义模型结构
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        # 定义模型的输入维度、隐藏层维度、隐藏层层数、输出维度
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 定义双向LSTM层，输入维度为input_size，隐藏层维度为hidden_size，隐藏层层数为num_layers，batch_first=True表示输入数据的维度为(batch_size, seq_length, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # 定义全连接层，输入维度为hidden_size * 2，因为是双向的，所以要乘以2，输出维度为output_size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # 初始化隐藏层和细胞状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # 初始化细胞状态
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out


# 定义训练函数
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    # 记录训练集和验证集的loss
    train_loss = []
    train_acc = []
    bp=tqdm(range(epochs))
    for epoch in bp:
        # 训练集
        losses = 0
        model.train()
        # 遍历训练集
        for i, (inputs, labels) in enumerate(train_loader):
            #   将数据移动到GPU上
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()
        #       print(f'Epoch [{epoch}/{num_epochs}], Loss: {round(losses / len(train_loader), 4)}')
        bp.set_description(f'Epoch [{epoch}/{num_epochs}], Loss: {round(losses / len(bp), 4)}')
        train_loss.append(losses / len(bp))

        correct = 0
        total = 0
        model.eval()
        # 验证集
        # 遍历验证集
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # 统计预测正确的数量
                correct += (predicted == labels).sum().item()
        val_acc = 100.0 * correct / total
        train_acc.append(val_acc)
#         print('Accuracy of the network on the validation data: %d %%' % (val_acc))
    # torch.save(model.state_dict(),'LSTM_model.pth')
    return train_loss, train_acc


# 定义验证函数
def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_acc = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # 都转化为float类型
            correct += (predicted == labels).sum().item()
    val_acc = 100.0 * correct / total
            
            
            # loss = criterion(outputs, labels)
            # val_acc += torch.sum(torch.round(outputs) == labels).item()
        # val_acc /= len(val_loader)
    print(val_acc)
    return val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default='0', help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=40, help='number of epochs to train (default: 100)')
    parser.add_argument('--num_classes', type=int, default=10, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
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
    # 注释掉的是用于多GPU训练的代码
    X_data, y_data, map_labels, random_data, random_y = read_data()
    # X_data, y_data, map_labels, random_data, random_y = read_data()
    train_data, val_test_data, train_y, val_test_y = train_test_split(X_data, y_data, test_size=0.4,
                                                                      random_state=42)
    val_data, test_data, val_y, test_y = train_test_split(val_test_data, val_test_y, test_size=0.5, random_state=0)
    train_loader = create_data_loader(train_data, train_y, batch_size=batch_size, num_workers=num_workers)
    val_loader = create_data_loader(val_data, val_y, batch_size=batch_size, num_workers=num_workers)
    test_loader = create_data_loader(test_data, test_y, batch_size=batch_size, num_workers=num_workers)
    # 定义模型，损失函数，优化器，学习率，训练次数，是否使用GPU
    model = BiLSTM(5, 256, 1, args.num_classes).to(device)
    model.load_state_dict(torch.load('LSTM_model_fu.pth'))
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # losses, acc = train(model, train_loader, val_loader, criterion, optimizer, device, epochs=args.num_epochs)
    val_acc = evaluate(model, test_loader, criterion, device)
    print(val_acc)

    # 绘制Loss曲线
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over time')
    plt.show()
    # 绘制Accuracy曲线
    plt.plot(range(1, len(acc) + 1), acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over time')
    plt.show()
    # 统计结果
    fig, ax = plt.subplots(10, 5, figsize=(20, 35))
    accuracy_dict = {}
    for index, (x, y) in enumerate(zip(random_data, random_y)):
        loader = create_data_loader(x, y, batch_size=batch_size, num_workers=num_workers)
        with torch.no_grad():
            total_correct = 0
            total_samples = 0
            for data, labels in loader:
                labels = labels.long().to(device)
                outputs = model(data.to(device))
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                # 在子图的右侧添加预测标签
                for j, ig in enumerate(data):
                    if j >= 5:
                        break
                    predicted_label = map_labels[predicted[j].item()]
                    ax[index][j] = drawing(ig, ax[index][j])
                    color = 'black' if labels[j] == predicted[j] else 'red'
                    ax[index][j].set_title(f"{predicted_label}", fontsize=14, color=color)
                title = ax[index][0].set_ylabel(f"{map_labels[index]}", fontsize=14, color='blue')
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
