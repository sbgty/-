# 实验三：图像拼接

## 一、实验目的
学会Opencv的基本使用方法，利用OpenCV等计算机库对图像进行特征点提取，图像拼接等操作。

## 二、实验内容
### 2.1 导入实验所需的依赖包
<div align="left"><img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/5e0dab4c-1fcf-4fbe-9e57-9719606cdc1e" /><div>

### 2.2 图像的读取与预处理
<div align="left"><img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/095e6e3d-493c-4fe7-ad8f-7b1cd082e476" /><div>

### 2.3 特征匹配
<div align="left"><img width="800" height="200" alt="image" src="https://github.com/user-attachments/assets/b2bd6e49-0519-4a82-85fb-860143111f3d" />
<div>

### 2.4 筛选优质匹配点
<div align="left"><img width="800" height="200" alt="{C88100A2-2CC2-4FDC-9290-4E12695A2952}" src="https://github.com/user-attachments/assets/d617d665-fb78-4ee0-8a13-358cea960bd2" />
<div>

### 2.5 可视化匹配结果
<div align="left"><img width="800" height="200" alt="{DF9DDACF-ED25-4331-AE51-7B6167199EA2}" src="https://github.com/user-attachments/assets/26543aa1-dfb9-4ffc-9665-2348a5927995" />
<div>

### 2.6 计算单应性矩阵
<div align="left"><img width="800" height="200" alt="{F7E2D1EF-077F-4A0F-A976-51FA6A36BFA4}" src="https://github.com/user-attachments/assets/6b5bb380-69e8-46ea-8749-38f2e81cde67" />
<div>

### 2.7 计算拼接画布尺寸
<div align="left"><img width="800" height="200" alt="{23CC7EDF-09E5-4C37-BBA9-965A334D43B5}" src="https://github.com/user-attachments/assets/c0612d0c-7057-483f-ab37-dce29ba35bbf" />
<div>

### 2.8 图像的拼接
<div align="left"><img width="800" height="200" alt="{A19C2BE3-9E8B-4BF3-992B-5328461A1F63}" src="https://github.com/user-attachments/assets/ff771232-8f69-4342-b294-7cac04126465" />
<div>
  
### 2.9 图像可视化  
<div align="left"><img width="800" height="200" alt="{CBE1948A-6BA1-4B93-A838-D2BE107BB6B6}" src="https://github.com/user-attachments/assets/8afcf90f-6371-43cb-b318-691317e81204" />
<div>
  
### 可视化结果
<div align="left"><img width="800" height="500" alt="{D90B905B-EEBF-4B41-91A8-E9DC5815BACC}" src="https://github.com/user-attachments/assets/0d4ee044-fd92-4c31-bd42-269c2547ccc7" />
<div>

## 三、实验结果与分析

### 1、图像的预处理与特征点提取
通过OpenCV读取原始图像进行灰度图像转化方便处理

采用SIFT算法提取两张图中的特征点

### 2、选择并筛选匹配点 
设置FLANN匹配器，并使用K近邻算法进行特征匹配

使用Lowe's比率筛选优质匹配点，保留空间分布均匀的可靠匹配点

### 3、图像拼接
使用RANSAC算法计算单应性矩阵

计算拼接画布

## 四、实验小结
本次实验完成了图像拼接的操作实验，熟悉了SIFT算法的使用，FLANN匹配器的建立以及多个关于图像拼接的操作，进一步加深了对于图像处理的理解程度，对于图像拼接的原理。
<div align="left"><div>
<div align="left"><div>
