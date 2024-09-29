# trainModel0.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import data_visualization
from torch.utils.data import DataLoader, Dataset
# import lpips
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim
# from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
import arithmeticcoding
import datetime
import random
from torch.utils.data import Subset
from CombinedLoss import CombinedLoss
import model2 as model
from test1 import process_image, quantize, dequantize, arithmetic_encode, arithmetic_decode

# 确保模型保存文件夹和重构图片文件夹存在
os.makedirs('model_pt', exist_ok=True)
os.makedirs('reconstruct_pic', exist_ok=True)

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 目标相对路径
relative_path = '../images'

# 计算目标绝对路径
dataset_dir = os.path.abspath(os.path.join(current_dir, relative_path))
#dataset_dir = '../images'     #     flickr30k-images
test_pic = 'test.png'  #test.png
binFile = 'temp_encoded.bin'
N, M = 256, 192
num_epochs = 10
# num_epochs = 2

# CUDA
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU ,")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((N, N)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(10),  # 随机旋转角度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机调整亮度、对比度、饱和度和色调
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 数据加载
dataset = data_visualization.CustomDataset(directory=dataset_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 随机抽取小样本训练
# Step 1: 数据集抽样
total_samples = len(dataset)  # 获取数据集总样本数
random_indices = random.sample(range(total_samples), 100)  # 随机抽取3000个索引
subset_dataset = Subset(dataset, random_indices)  # 创建包含抽取样本的子集
# Step 2: 创建新的DataLoader
subset_dataloader = DataLoader(subset_dataset, batch_size=32, shuffle=True)

# 定义模型
my_model = model.ScaleHyperprior(N, M).to(device)
my_loss = CombinedLoss().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001)  #0.0001
'''增加学习率衰减策略'''
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


# 训练模型
train_loss_history = []
best_score = -float('inf')
best_model_path = 'model_pt/model_best.pth'

with open('loss_history.txt', 'w') as f:
    for epoch in range(num_epochs):
        print(epoch)
        total_loss = 0
        for index, data in enumerate(subset_dataloader):      # 自己训练时选择完整的数据：subset_dataloader
            # print('data:', index)
            data = data.to(device)
            # print(data.size())
            # data = F.interpolate(data, size=(N, N))

            # 前向传播
            recon, latent = my_model(data)
            # print(recon)
            # print(latent)
            # 量化、编码、解码、反量化
            quantized_latents, scale, min_val = quantize(latent)
            # quantized_flat = quantized_latents.numpy().flatten()#.cpu()
            quantized_flat = quantized_latents.flatten()#.cpu()
            # print(quantized_flat)
            arithmetic_encode(quantized_flat, binFile)
            # print('decoded')
            decoded_latents = arithmetic_decode(binFile)
            # print(decoded_latents)
            # 确保解码的是NumPy array
            decoded_latents = np.array(decoded_latents)
            if len(decoded_latents) < quantized_latents.numel():
                decoded_latents = np.pad(decoded_latents, (0, quantized_latents.numel() - len(decoded_latents)),
                                         mode='constant')
            elif len(decoded_latents) > quantized_latents.numel():
                decoded_latents = decoded_latents[:quantized_latents.numel()]
            decoded_latents = torch.tensor(decoded_latents.reshape(quantized_latents.shape)).float().to(device)
            decoded_latents = dequantize(decoded_latents, scale, min_val).to(device)

            # 通过解码部分还原图像
            recon = my_model.ha.decoder(decoded_latents)
            # print(recon)
            # print(data)
            # print(latent)
            # assert recon.shape == data.shape, f"output sharp {recon.shape} and target sharp {data.shape} not "
            loss = my_loss(data, recon, latent)
            total_loss += loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        train_loss_history.append(avg_loss)
        '''每个epoch之后调用学习率调度器'''
        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

        # 将每个epoch的loss写入文件
        f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}\n')

        '''从第10个epoch之后，每个epoch调用一次process_image进行test'''
        if epoch + 1 >= 10:
        # if epoch + 1 >= 1:
            model_path = f'model_pt/model_epoch_{epoch + 1}.pth'
            torch.save(my_model.state_dict(), model_path)
            psnr_value, ssim_value, lpips_value = process_image(test_pic, model_path, 'encoded_3.bin',
                                                            f'reconstruct_pic/reconstructed_epoch_{epoch + 1}.png')

            # 计算综合质量指标
            score = 0.4 * psnr_value + 0.4 * ssim_value - 0.2 * lpips_value

            # 保存质量最好的模型
            if score > best_score:
                best_score = score
                best_model_path = model_path
                torch.save(my_model.state_dict(), 'model_pt/model_best.pth')
                print(f'New best model saved with score: {score:.4f} at epoch {epoch + 1}')

# save the final model
torch.save(my_model.state_dict(), 'model_pt/model_final.pth')

# Draw and save loss history
data_visualization.plot_and_save_loss(train_loss_history, 'model_hyperprior_50_conv_2.png')
