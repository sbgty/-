# 实验七、残差网络

## 一、实验目的
掌握 ResNet 残差网络在图像分类任务中的基本用法，使用预训练 ResNet-18 进行迁移学习，加快收敛并提升精度，并评估分类性能，输出准确率以及混淆矩阵。

## 二、实验内容

### 2.1、导入实验所需的库
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    from sklearn.metrics import confusion_matrix
    import numpy as np

### 2.2、配置基本参数以及使用硬件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 128
    EPOCHS = 5
    LR = 1e-3   
    NUM_CLASSES = 10

### 2.3、
