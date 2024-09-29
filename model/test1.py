import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import data_visualization
from torch.utils.data import DataLoader, Dataset
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
import arithmeticcoding
import datetime
import random
from torch.utils.data import Subset
from closs import CLoss
import model2 as model  # 修改后的网络

# 定义编码和解码过程中使用的函数
def process_image(test_image_path='test.png', model_path='model_hyperprior_10_conv_1024_7.pth', binFile='encoded_3.bin',
                  output_image_path='model_hyperprior_10_conv_1024_7.png', image_size=(256, 256)):
    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found at {test_image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")


    # CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试图像
    test_image = Image.open(test_image_path).convert('RGB')
    test_image = transform(test_image)
    test_image = test_image.unsqueeze(0).to(device)

    # 加载模型
    my_model = model.ScaleHyperprior(image_size, 192).to(device)  # 使用动态尺寸
    my_model.load_state_dict(torch.load(model_path, map_location=device))
    my_model.eval()

    # 推理和量化
    with torch.no_grad():
        recon, latent = my_model(test_image)  # 前向传播，获取潜在表示
        quantized_latents, scale, min_val = quantize(latent)  # 量化
        # quantized_flat = quantized_latents.numpy().flatten()  # 展平量化后的数据 .cpu()
        quantized_flat = quantized_latents.flatten()  # 展平量化后的数据 .cpu()

        # 编码
        arithmetic_encode(quantized_flat, binFile)

        # 解码
        decoded_latents = arithmetic_decode(binFile)

        # 确保解码的是NumPy array
        decoded_latents = np.array(decoded_latents)

        if len(decoded_latents) < quantized_latents.numel():
            decoded_latents = np.pad(decoded_latents, (0, quantized_latents.numel() - len(decoded_latents)), mode='constant')
        elif len(decoded_latents) > quantized_latents.numel():
            decoded_latents = decoded_latents[:quantized_latents.numel()]

        decoded_latents = torch.tensor(decoded_latents.reshape(quantized_latents.shape)).float().to(device)
        decoded_latents = dequantize(decoded_latents, scale, min_val).to(device)    # 反量化
        recon = my_model.ha.decoder(decoded_latents)  # 通过模型的解码部分解码

    # 保存重建图像
    recon_image = transforms.ToPILImage()(recon.squeeze(0))
    recon_image = recon_image.resize((256, 256))
    recon_image.save(output_image_path)

    # 计算评估指标
    original = Image.open(test_image_path).convert('RGB')
    reconstructed = Image.open(output_image_path).convert('RGB')
    original = original.resize(image_size)
    reconstructed = reconstructed.resize(image_size)

    psnr_value = calculate_psnr(original, reconstructed)
    ssim_value = calculate_ssim(original, reconstructed)
    lpips_value = calculate_lpips(original, reconstructed)

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    with open("evaluation.txt", "a") as f:
        f.write(
            f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f} ({timestamp}), Model Name：{model_path}\n")
    print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}")

    return psnr_value, ssim_value, lpips_value

def quantize(latents, levels=256):
    min_val, max_val = latents.min(), latents.max()
    scale = (max_val - min_val)
    quantized = torch.round((latents - min_val) / scale * (levels - 1)).to(torch.uint8)
    return quantized, scale, min_val


def dequantize(quantized, scale, min_val):
    return quantized.float() * scale + min_val


def arithmetic_encode(data, binFile):
    freq = arithmeticcoding.FrequencyTable([1] * 256 + [100])
    # print(1111)
    bitout = arithmeticcoding.BitOutputStream(open(binFile, "wb"))
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)
    # print(222)
    # print(enc)
    # enc.write(freq, torch.tensor(255, device='cuda:0', dtype=torch.uint8).item())
    # print(len(data))
    for symbol in data:
        # print(symbol)
        enc.write(freq, symbol.item())

    # print(0000)
    enc.write(freq, 256)
    enc.finish()
    bitout.close()


def arithmetic_decode(binFile):
    bitin = arithmeticcoding.BitInputStream(open(binFile, "rb"))
    freq = arithmeticcoding.FrequencyTable([1] * 256 + [100])
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
    decoded = []
    while True:
        symbol = dec.decode(freq)
        if symbol == 256:
            break
        decoded.append(symbol)
    bitin.close()
    return decoded

'''
def calculate_psnr(original, reconstructed):
    original = original.resize(reconstructed.size, Image.LANCZOS)
    original_np = np.array(original)
    reconstructed_np = np.array(reconstructed)
    return psnr(original_np, reconstructed_np, data_range=255)


def calculate_ssim(original, reconstructed):
    original = original.resize(reconstructed.size, Image.LANCZOS)
    original_np = np.array(original)
    reconstructed_np = np.array(reconstructed)
    min_dim = min(original_np.shape[0], original_np.shape[1])
    win_size = min(7, min_dim)
    return ssim(original_np, reconstructed_np, data_range=255, multichannel=True, win_size=win_size, channel_axis=-1)


def calculate_lpips(original, reconstructed):
    original = original.resize(reconstructed.size, Image.LANCZOS)
    original_tensor = transforms.ToTensor()(original).unsqueeze(0)
    reconstructed_tensor = transforms.ToTensor()(reconstructed).unsqueeze(0)
    lpips_value = lpips_fn(original_tensor, reconstructed_tensor)
    return lpips_value.item()
'''
def calculate_psnr(original, reconstructed):
    reconstructed = resize_if_needed(reconstructed, original.size)
    original_np = np.array(original)
    reconstructed_np = np.array(reconstructed)
    psnr_value = psnr(original_np, reconstructed_np, data_range=255)
    return psnr_value


def calculate_ssim(original, reconstructed):
    reconstructed = resize_if_needed(reconstructed, original.size)
    original_np = np.array(original)
    reconstructed_np = np.array(reconstructed)
    win_size = min(3, original_np.shape[0], original_np.shape[1])
    ssim_value = ssim(original_np, reconstructed_np, data_range=255, multichannel=True, win_size=win_size)
    return ssim_value


def calculate_lpips(original, reconstructed):
    reconstructed = resize_if_needed(reconstructed, original.size)
    original_tensor = transforms.ToTensor()(original).unsqueeze(0)
    reconstructed_tensor = transforms.ToTensor()(reconstructed).unsqueeze(0)
    # lpips_fn = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_fn = lpips.LPIPS(net='alex').to('cpu')
    lpips_value = lpips_fn(original_tensor, reconstructed_tensor)
    return lpips_value.item()

def resize_if_needed(image, size):
    width, height = image.size
    target_width, target_height = size
    if width < target_width or height < target_height:
        return image.resize(size, Image.LANCZOS)
    return image


if __name__ == "__main__":
    psnr_value, ssim_value, lpips_value = process_image(image_size=(256, 256))
    print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}")
