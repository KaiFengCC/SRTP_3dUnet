import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import *
from Net import *
from utils import *
import matplotlib.pyplot as plt

# ����CUDA�豸
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")

# ��������·��
dataPath = r'./srtpDataset'

# �������ݼ�
trainDataset = ImageDataSet(dataPath)

# �������ݼ�����
trainLoader = DataLoader(trainDataset, batch_size=10, shuffle=True)  

# ��������ģ��
net = UNet().to(device)

# ������ʧ�������Ż���
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(net.parameters(), lr=0.001)

# ����һ��ͼ�κ�������ͼ��һ��������ʧ��һ�������ݶ�
fig, (ax1, ax2) = plt.subplots(2, 1)
# ���ڱ�����ʧ���ݶȵ��б�
losses = []
gradients = []
# ���ڱ��������ʧ
best_loss = float('inf')

# ѵ������
for epoch in range(10):  # ����������������ѵ��������
    epoch_loss=0
    for i, data in enumerate(trainLoader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # ǰ�򴫲�
        outputs = net(inputs)

        # ������ʧ
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        # ���򴫲����Ż�
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ������ʧ���ݶ�
        losses.append(loss.item())
        gradients.append(net.conv1.weight.grad.abs().mean().item())
        # ������ʧ���ݶȵ�ͼ��
        ax1.clear()
        ax1.plot(losses)
        ax1.set_title('Loss over time')
        ax2.clear()
        ax2.plot(gradients)
        ax2.set_title('Gradient over time')
        plt.pause(0.01)


        # ��ӡͳ����Ϣ
        if i % 3 == 0:  # ÿ100�����δ�ӡһ��
            print(f'Epoch {epoch+1}/{10}, Step {i+1}/{len(trainLoader)}, Loss: {loss.item()}')

    # ����ƽ����ʧ
    avg_loss = epoch_loss / len(trainLoader)
    # ���ƽ����ʧ���ͣ�����ģ��
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(net.state_dict(), f'best_model_epoch_{epoch+1}.pth')

print('Finished Training')

