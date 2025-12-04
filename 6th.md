# 实验六、图像分类

## 一、实验目的
学会如何使用 Inception-v3 模型对数据集进行图像分类。学习如何将 Fashion-MNIST 数据集适配到 Inception-v3 输入规格

## 二、实验内容

### 2.1 导入实验所需的库及配置环境
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import Subset
    from torchsummary import summary
    import numpy as np
      
### 2.2 硬件设备检测与基本参数设置 
    batch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
设置批处理大小为 16，平衡内存使用和训练稳定性，自动检测可用设备（优先使用 GPU，回退到 CPU）。

### 2.3 数据预处理与增强
    transform = transforms.Compose([
    transforms.Resize(299),  # 调整图像大小为299×299
    transforms.Grayscale(num_output_channels=3),  # 单通道转三通道
    transforms.ToTensor()  # 转为Tensor并归一化到[0,1]
    ])
将28×28的Fashion-MNIST图像上采样到299×299，复制灰度通道生成3通道RGB图像（适配Inception-v3），将PIL图像转为PyTorch张量，并自动归一化像素值。

### 2.4 数据集加载与采样
    # 2.4.1 加载完整数据集
    train_full = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    test_full = datasets.FashionMNIST('data', train=False, download=True, transform=transform)

    # 2.4.2 数据采样（使用1%数据）
    n = 100  # 采样比例参数
    rng = np.random.default_rng(42)  # 固定随机种子确保可重复性
    train_idx = rng.choice(len(train_full), len(train_full) // n, replace=False)
    test_idx = rng.choice(len(test_full), len(test_full) // n, replace=False)
下载并加载Fashion-MNIST训练集（60,000张）和测试集（10,000张），仅使用1%数据（600张训练，100张测试），加快实验速度，并固定随机种子42确保每次实验采样一致。

### 2.5 创建数据加载器
    train_loader = torch.utils.data.DataLoader(Subset(train_full, train_idx),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Subset(test_full, test_idx),
        batch_size=batch_size, shuffle=False)
创建基于采样索引构建训练和测试子集，将数据划分为大小为16的批次，每个epoch随机打乱训练数据顺序，防止模型记忆顺序，而评估时不打乱，便于结果分析。

### 2.6 预训练模型加载，修改与优化器配置
    # 6.1 加载预训练Inception-v3模型
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

    # 6.2 修改分类层
    model.fc = nn.Linear(model.fc.in_features, 10)

    # 6.3 模型设备转移
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
使用在ImageNet上预训练的Inception-v3，将最后的全连接层从1000类（ImageNet）改为10类（Fashion-MNIST），使用Adam优化器，结合动量和自适应学习率

### 2.7 训练循环实现
    epochs = 10
    accs, losses = [], []  # 记录性能指标

    for epoch in range(epochs):
        # 9.1 训练模式设置
        model.train()
    
        # 9.2 批次训练
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # 数据转移
            optimizer.zero_grad()  # 梯度清零
            out = model(x).logits  # 前向传播（获取主分类器输出）
            loss = F.cross_entropy(out, y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 参数更新
循环训练模型，通过数据转移，确保数据和模型在同一设备，前向传播计算模型预测值，计算损失使用交叉熵损失衡量预测与真实标签差异，反向传播计算参数梯度，参数更新将根据梯度更新模型参数。

### 2.8 模型评估与性能记录
    # 10.1 评估模式设置
    model.eval()
    correct, total_loss = 0, 0.
    
    # 10.2 测试集推理（无梯度计算）
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)  # 前向传播
            total_loss += F.cross_entropy(out, y).item()  # 累计损失
            correct += (out.argmax(1) == y).sum().item()  # 统计正确预测数
    
    # 10.3 性能指标计算
    acc = correct / len(test_loader.dataset)  # 准确率
    avg_loss = total_loss / len(test_loader)  # 平均损失
    accs.append(acc)  # 记录准确率
    losses.append(avg_loss)  # 记录损失
    
    # 10.4 进度输出
    print(f'epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}')
统计性能指标，将批次损失累加，统计正确预测样本，计算准确率以及平均损失，保存每个epoch的性能指标，并将其可视化输出。

## 三、实验结果与分析
<div ailgn="left"><img width="600" height="380" alt="image" src="https://github.com/user-attachments/assets/82efad3e-b280-497c-bd23-5ca3871aa6f6" />
<div>
模型训练所得结果尚可，最高准确率达到84%，损失达到0.4354，但所得损失以及准确率波动较大，推测是训练量以及训练轮次太少。

## 四、实验小结
本次实验完成了通过Inception-v3 模型对数据集进行图像分类，初步理解图像分类的工作流程，加深了对于课上内容的印象，为后续学习打下了扎实的基础。
