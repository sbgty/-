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

### 2.3、配置一个 epoch：模型前向、计算损失、反向传播、更新参数
    def train_one_epoch(model, train_loader, optimizer, criterion):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 清空上一次反向传播残留的梯度
            optimizer.zero_grad()
            # 前向传播：ResNet 输出每类的分数（未归一化），形状 [B, 10]
            logits = model(x)
            # 用交叉熵损失计算预测与真实标签差距
            loss = criterion(logits, y)
            # 反向传播：自动求梯度
            loss.backward()
            # 按优化器规则更新参数
            optimizer.step()

            # 统计指标
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item() 
            total += x.size(0)
        return total_loss / total, correct / total

## 2.4、测试评估
    @torch.no_grad()    
    def evaluate(model, test_loader):
        model.eval()
        correct, total = 0, 0
        all_pred, all_true = [], []
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())

        all_pred = np.concatenate(all_pred)
        all_true = np.concatenate(all_true)
        cm = confusion_matrix(all_true, all_pred)
        return correct / total, cm

## 2.5、主函数配置
    def main():
        transform_train = transforms.Compose([
            # 需改输入为 224×224 符合 ResNet 预训练通常输入
            transforms.Resize(224),
            # 训练时随机水平翻转，增加数据多样性
            transforms.RandomHorizontalFlip(),
            # 把 PIL 图片转成 tensor，并把像素值缩放到 [0,1]
            transforms.ToTensor(),
            # 使用 ImageNet 的均值方差归一化，匹配预训练模型习惯
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # 训练集不做随机反转测试
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # 检测 CIFAR-10 数据集
        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        test_set  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

        # 构建 DataLoader：按 batch 提供数据
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # 加载 ResNet-18 预训练模型并替换分类层
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model = model.to(device)

        # 冻结主干，仅训练最后的分类层（简化迁移学习）
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

        # 定义损失函数与优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

        # 训练与测试循环：每轮训练后做一次评估
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            test_acc, cm = evaluate(model, test_loader)
            print(f"Epoch {epoch}/{EPOCHS} | loss={train_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

        print("\nConfusion Matrix:\n", cm)

    if __name__ == "__main__":
        main()

## 三、实验结果和分析
<div ailgn="left"><img width="550" height="400" alt="image" src="https://github.com/user-attachments/assets/d8ab54b9-0cf9-4525-a58e-1fc2ddad684d" />
<div>
根据输出可知，损失正常下降，说明训练过程正常不存在错误，同时准确率稳定上升的同时保持较高的水平，训练以及测试都维持较高的水准；经分析混淆矩阵可以得出，交通工具率样本识别准确率较高，而猫、狗等外观较近的动物类别存在较高的混淆现象。总体而言，实验成功验证了残差网络在计算机视觉中的有效性。

## 四、实验小结
本次实验成功实现了残差网络在图像分类任务中的基本用法，掌握了残差网络的基本用法，为后续学习打下基础。
