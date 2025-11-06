# 实验四 ：感知机

## 一、实验目的
学会感知机的搭建及使用，利用PyTorch进行感知机的搭建运行实验。

## 二、实验内容

### 2.1 导入实验所需的依赖包
<div align="left"><img width="550" height="150" alt="{0DC66A39-4568-407E-8211-752DE54AD33D}" src="https://github.com/user-attachments/assets/6d63778d-acaa-4c33-a4a9-4f7a48874bec" /><div>

### 2.2 配置超参数以及设备
<div align="left"><img width="550" height="50" alt="{70C70CE2-A6C2-429A-BCFD-E20509ED3DE5}" src="https://github.com/user-attachments/assets/eb443574-bb5a-4112-ba86-28db649518d3" /><div>
设置批次大小以及设定使用CPU训练

### 2.3 数据的加载与预处理以及数据形状分析
<div align="left"><img width="550" height="200" alt="{F05016B2-26AA-4428-BEF8-F07BA298D8EE}" src="https://github.com/user-attachments/assets/848607dd-383f-4702-aec7-8ee30bec5bc4" />
<div>

### 2.4 神经网络模型设计
<div align="left"><img width="550" height="200" alt="{3D776FED-D8EF-4839-9B8E-D4A6C21C4F3B}" src="https://github.com/user-attachments/assets/8dd7d355-3516-4426-b94c-57b271453c59" />
<div>

### 2.5 模型实例化以及配置优化器
<div align="left"><img width="550" height="50" alt="{02333801-1FEC-46C0-9CC7-50124CFF8FF3}" src="https://github.com/user-attachments/assets/f2d1ec3d-6d83-496e-803c-545a77f902e0" />
<div>

### 2.6 训练及测试
<div align="left"><img width="550" height="350" alt="{499CD89E-8B12-4C3B-BE93-433B79AC9D2C}" src="https://github.com/user-attachments/assets/29b87519-919c-4e0b-bc8c-6a8b38daf906" />
<div>

### 2.7 性能评估和输出
<div align="left"><img width="550" height="80" alt="{ECCBB81A-9B34-48ED-85EB-57F807EF97EF}" src="https://github.com/user-attachments/assets/f05ea8f8-120c-47fb-b942-2724c104b484" />
<div>

### 2.8 输出结果与分析
<div align="left"><img width="550" height="300" alt="{71558B12-C43A-446A-BA74-2844A1869137}" src="https://github.com/user-attachments/assets/170ec808-1f9f-4516-9c68-8d7751121bc1" />
<div>
最终准确率: 90.36%  最终损失: 0.3451  训练稳定性: 整个训练过程稳定，没有出现震荡  


## 三、实验结果与分析
### 1、超参数设置
选择2048的大批次训练，梯度更加稳定，训练速度更快。

使用SGD优化器，设置较高学习率。

### 2、数据预处理
使用标准MNIST手写数字数据集，分为训练集和测试集。

配置批量数据加载，训练集启用shuffle增强随机性，测试集便于评估。

### 3、模型设计
设计两层简单感知机，简单，便捷化处理。

## 4、实验小结
通过本次实验，掌握了感知机的构建以及使用方式，加深了对于计算机视觉的理解程度，巩固了课上的内容。为后续的学习打下基础。

