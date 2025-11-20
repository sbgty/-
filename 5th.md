# 实验五、神经网络

## 一、实验目的
学会神经网络的搭建与使用，使用Pytorch进行神经网络的相关实验。

## 二、实验内容

### 2.1 导入实验所需的包及搭建环境
<div align="left"><img width="800" height="350" alt="image" src="https://github.com/user-attachments/assets/4598d3a9-78ff-4d6b-8fa0-71a1f08330c9" />
<div>

### 2.2 数据加载和预处理
<div align="left"><img width="800" height="350" alt="image" src="https://github.com/user-attachments/assets/38617184-de05-470d-be43-2452d550c27d" />
<div>

### 2.3 定义LeNet神经网络模型
<div align="left"><img width="800" height="350" alt="image" src="https://github.com/user-attachments/assets/cd9286b8-af34-4153-8433-a0ee9b9fae64" />
<div>

### 2.4 前向传播过程
<div align="left"><img width="800" height="700" alt="image" src="https://github.com/user-attachments/assets/22437990-092a-4c4e-96b8-2671b57f9ad1" />
<div>

### 2.5 模型训练循环
<div align="left"><img width="800" height="350" alt="image" src="https://github.com/user-attachments/assets/e14930a9-4c8b-452c-88f3-8113a5a391fd" />
<div>

### 2.6 模型的测试和评估
<div align="left"><img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/9ead0384-bd63-4968-bb6a-83a9dd955586" />
<div>

### 2.7 特征图可视化
<div align="left"><img width="800" height="700" alt="image" src="https://github.com/user-attachments/assets/b8c0a5ab-a9f5-4ffd-872c-a2507df1d3d5" />
<div>

### 实验结果
<div align="left"><img width="800" height="700" alt="image" src="https://github.com/user-attachments/assets/2ca15500-ae87-4bb7-a258-ba4cdd9f0796" />
<div>

## 三、实验结果与分析
### 1、Letnet神经网络模型
    通道数由1→6→16逐步增加，符合特征提取的渐进性原则。
    通过设置较小的卷积核便于捕捉局部特征。

### 2、前向传播过程
    通过第一和第二卷积块，分别执行提取图像基础特征，降维并增强特征鲁棒性以及组合基础特征形成复杂模式并进一步压缩特征尺寸。
    完成特征展平，将二位特征图转换为一维向量，为分类做准备。
    构建全连接层，整合所有特征信息，逐步提炼最具判别性的特征。

### 3、训练模型
    经过数据迁移，梯度清零，前向传播，计算损失，反向传播以及参数更新几步后，重复训练，多次悬链调整模型，使其能够准确辨别手写数字。

## 四、实验小结
通过本次实验掌握了如何构建与使用神经网络，进一步加深了对于课上内容的理解，巩固了课上的知识，为后续学习打下了扎实的基础。
