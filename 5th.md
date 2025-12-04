# 实验五、神经网络

## 一、实验目的
学会神经网络的搭建与使用，使用Pytorch进行神经网络的相关实验。

## 二、实验内容

### 2.1 导入实验所需的包及搭建环境
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt

    batch_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### 2.2 数据加载和预处理
    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True
    )
加载数据集以及自动预处理数据集。

### 2.3 定义LeNet神经网络模型
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 输入通道1，输出通道6，卷积核5x5
            self.conv2 = nn.Conv2d(6, 16, 5)  # 输入通道6，输出通道16，卷积核5x5
            self.fc1 = nn.Linear(16  * 5 * 5, 128)  # 全连接层，输入16*5*5=400，输出128
            self.fc2 = nn.Linear(128, 84)  # 全连接层，输入128，输出84
            self.clf = nn.Linear(84, 10)  # 分类层，输入84，输出10
        def forward(self, x):
            # conv1
            x = self.conv1(x)
            # 激活函数sigmoid()
            x = F.sigmoid(x)
            # 平均池化层，kernel=2x2，步长2
            x = F.avg_pool2d(x, kernel_size=2, stride=2)

            # conv2
            x = self.conv2(x)
            # 激活函数sigmoid()
            x = F.sigmoid(x)
            # 平均池化层，2x2，步长2
            x = F.avg_pool2d(x, kernel_size=2, stride=2)

            # 展开，从第1维开始展开
            x = x.view(x.size(0), -1)
    
            # 全连接层1
            x = self.fc1(x)
            # 激活函数sigmoid()
            x = F.sigmoid(x)

            # 全连接层2
            x = self.fc2(x)
            # 激活函数sigmoid()
            x = F.sigmoid(x)

            # 分类层
            x = self.clf(x)
            return x
配置LeNet网络模型。

### 2.4 模型训练循环与评估
    epochs = 30
    accs, losses = [], []

    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
    
        correct = 0
        testloss = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloader):
                x, y = x.to(device), y.to(device)
                out = model(x)
                testloss += F.cross_entropy(out, y).item()
                pred = out.max(dim=1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()

        acc = correct / len(testloader.dataset)
        testloss = testloss / (batch_idx + 1)
        accs.append(acc)
        losses.append(testloss)
        print('epoch:{}, loss:{:.4f}, acc:{:.4f}'.format(epoch, testloss, acc))
训练模型，同时在每次训练后输出每次训练的损失以及准确率。
 


### 2.5 特征图可视化
    # 可视化特征图
    feature1 = F.sigmoid(model.conv1(x))
    feature1 = F.avg_pool2d(feature1, kernel_size=2, stride=2)
    feature2 = F.sigmoid(model.conv2(feature1))
    feature2 = F.avg_pool2d(feature2, kernel_size=2, stride=2)

    n = 5
    img = x.detach().cpu().numpy()[:n]
    feature_map1 = feature1.detach().cpu().numpy()[:n]
    feature_map2 = feature2.detach().cpu().numpy()[:n]

    fig, ax = plt.subplots(3, n, figsize=(10, 10))
    for i in range(n):
        ax[0, i].imshow(img[i].sum(0), cmap='gray')
        ax[0, i].axis('off')
        ax[1, i].imshow(feature_map1[i].sum(0), cmap='gray')
        ax[1, i].axis('off')
        ax[2, i].imshow(feature_map2[i].sum(0), cmap='gray')
        ax[2, i].axis('off')
    plt.show()




## 三、实验结果与分析

<div align="left"><img width="500" height="250" alt="image" src="https://github.com/user-attachments/assets/faa5653e-5fc5-434a-8a72-b582721e2f87" />
<div>
部分模型训练输出

由输出可见模型损失小，准确率高，可得模型有充分的可行性。


<div align="left"><img width="500" height="500" alt="image" src="https://github.com/user-attachments/assets/4d9ce284-2f9a-4a8f-88d4-c877b78eef1a" />

<div>
分别输出输入图，第一层卷积，sigmoid和池化后的特征图，第两层卷积，sigmoid和池化后的特征图。
    
## 四、实验小结
通过本次实验掌握了如何构建与使用神经网络，进一步加深了对于课上内容的理解，巩固了课上的知识，为后续学习打下了扎实的基础。
