import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import *
from Net import *
from utils import *
import matplotlib.pyplot as plt

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

# 设置数据路径
dataPath = r'./srtpDataset'

# 创建数据集
trainDataset = ImageDataSet(dataPath)

# 创建数据加载器
trainLoader = DataLoader(trainDataset, batch_size=10, shuffle=True)  

# 创建网络模型
net = UNet().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 创建一个图形和两个子图：一个用于损失，一个用于梯度
fig, (ax1, ax2) = plt.subplots(2, 1)
# 用于保存损失和梯度的列表
losses = []
gradients = []
# 用于保存最低损失
best_loss = float('inf')

# 训练网络
for epoch in range(10):  # 根据您的需求设置训练的轮数
    epoch_loss=0
    for i, data in enumerate(trainLoader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = net(inputs)

        # 计算损失
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 保存损失和梯度
        losses.append(loss.item())
        gradients.append(net.conv1.weight.grad.abs().mean().item())
        # 更新损失和梯度的图形
        ax1.clear()
        ax1.plot(losses)
        ax1.set_title('Loss over time')
        ax2.clear()
        ax2.plot(gradients)
        ax2.set_title('Gradient over time')
        plt.pause(0.01)


        # 打印统计信息
        if i % 3 == 0:  # 每100个批次打印一次
            print(f'Epoch {epoch+1}/{10}, Step {i+1}/{len(trainLoader)}, Loss: {loss.item()}')

    # 计算平均损失
    avg_loss = epoch_loss / len(trainLoader)
    # 如果平均损失更低，保存模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(net.state_dict(), f'best_model_epoch_{epoch+1}.pth')

print('Finished Training')

