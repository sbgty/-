# 实验二：图像增强  

## 一、实验目的
学会OpenCV的基本使用方法，利用OpenCV等计算机库对图像进行平滑、滤波等操作，实现图像增强。

## 二、实验内容
### 2.1 导入图像滤波相关的依赖包
<div align="left"><img width="500" height="182" alt="91eb457d595a3854aefe92ae6550202f" src="https://github.com/user-attachments/assets/cfe6a3e2-33fc-43b7-ab6e-43d7ac570fda" /><div>
  
### 2.2 读取原始图像并进行色彩空间转换
<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/30c84058-f268-4aa8-a104-9ce668e8f13c" /><div>
输出为
<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/6f92b955-dba6-47b9-8acb-7ef0b2e10422" /><div>
  
由于OpenCV的BGR格式与Matplotlib的RGB格式冲突，导致图像输出错误，产生红蓝通道互换的图像。
<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/f8601d5b-d263-41c7-9304-6376a3d46988" />
<div>
  
通过代码 cv2.COLOR_BGR2RGB 将BGR格式转换为RGB格式并测试输出图像，得到结果为原图。

<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/6d4a2288-970f-4c7b-9469-1c55ecb26f77" />
<div>
<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/e9788c46-9131-4792-8670-e99fb3502b3b" />
<div>

代码 cv2.COLOR_BGR2GRAY 将原始图的RGB通道映射为一味的灰度通道，测试灰度图。
<div align="left"><img width="500" height="480" alt="image" src="https://github.com/user-attachments/assets/63b6b506-7285-46ae-96e6-dea1575f1241" />
<div>
<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/f247839a-a918-429d-9621-d6f0708629ed" />
<div>
  
### 2.3 添加噪声
通过代码实现椒盐噪声以及高斯噪声的表现图，同时将原图与噪声图放在一起进行比较。

<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/91ad8bcd-8922-486e-9c01-32004c1cdebc" />
<div>
<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/464de9f4-e451-4f45-984b-e6d67bbfcb23" />
<div>
  
### 2.4 图像滤波
通过构建代码，对2.3中的椒盐噪声图分别进行均值滤波，中值滤波以及高斯滤波，将噪声图以及三张处理后的图片并列输出，进行对比。
<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/4061a7bf-579a-4201-ac5b-b07ca9da0ec2" />
<div>
<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/817d1b7e-e955-4af6-906f-cd7c7bcb2d11" />
<div>
  
### 2.5 手动实现均值滤波
#### 手动实现均值滤波
```
def mean_filter_manual(image, kernel_size=3):
    """
    手动实现均值滤波（支持彩色和灰度图像）

    参数:
    image: 输入图像(彩色或灰度图)
    kernel_size: 滤波核大小，默认为3x3

    返回:
    filtered_image: 滤波后的图像（保持原始色彩）
    """

    # 确保图像数据类型为uint8
    if image.dtype != np.uint8:
        if image.dtype in [np.float32, np.float64]:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

    # 判断是否为彩色图像
    if len(image.shape) == 3:
        height, width, channels = image.shape

        # 创建彩色输出图像
        filtered_image = np.zeros((height, width, channels), dtype=np.uint8)

        # 计算填充大小
        pad = kernel_size // 2

        # 对每个通道分别进行均值滤波
        for channel in range(channels):
            # 获取单通道图像
            single_channel = image[:, :, channel]

            # 边界填充
            padded_channel = cv2.copyMakeBorder(single_channel, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

            # 遍历每个像素
            for i in range(pad, height + pad):
                for j in range(pad, width + pad):
                    # 提取核区域
                    kernel_region = padded_channel[i - pad:i + pad + 1, j - pad:j + pad + 1]

                    # 计算平均值
                    mean_value = np.mean(kernel_region)

                    # 赋值到对应通道
                    filtered_image[i - pad, j - pad, channel] = mean_value

    else:
        # 灰度图像处理（原逻辑）
        height, width = image.shape
        filtered_image = np.zeros((height, width), dtype=np.uint8)
        pad = kernel_size // 2
        padded_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

        for i in range(pad, height + pad):
            for j in range(pad, width + pad):
                kernel_region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
                mean_value = np.mean(kernel_region)
                filtered_image[i - pad, j - pad] = mean_value

    return filtered_image

    manal_mean_3 = mean_filter_manual((sp_noise_img * 255).astype(np.uint8), 3)

    plt.figure(figsize=(10, 5))
    plt.subplot(1,2,1)
    plt.imshow(sp_noise_img,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(manal_mean_3,cmap='gray')
    plt.show()
```

<div align="left"><img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/1534dd8a-a99c-402e-868c-f173b32809f1" />
<div>

## 三、实验结果与分析

