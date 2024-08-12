

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
'''
class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, original, reconstructed, latent):
        mse = self.mse_loss(reconstructed, original)
        original_np = original.detach().cpu().numpy()
        reconstructed_np = reconstructed.detach().cpu().numpy()
        psnr_loss = -torch.tensor(psnr(original_np, reconstructed_np, data_range=1.0))
        min_side = min(original.size(2), original.size(3))
        win_size = max(3, min_side // 2 * 2 + 1)
        if min_side >= win_size:
            ssim_loss = -torch.tensor(
                ssim(original_np, reconstructed_np, data_range=1.0, channel_axis=-1, win_size=win_size))
        else:
            ssim_loss = torch.tensor(0.0)
        probs = F.softmax(latent, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return mse + 0.01 * psnr_loss + 0.01 * ssim_loss + 0.1 * entropy
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

class CLoss(nn.Module):
    def __init__(self):
        super(CLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lpips_loss = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, original, reconstructed, latent):
        mse = self.mse_loss(reconstructed, original)
        original_np = original.detach().cpu().numpy()
        reconstructed_np = reconstructed.detach().cpu().numpy()

        # 计算 PSNR 损失
        psnr_value = psnr(original_np, reconstructed_np, data_range=1.0)
        psnr_loss = torch.tensor(1.0 / psnr_value, dtype=torch.float32)

        # 计算 SSIM 损失
        min_side = min(original.size(2), original.size(3))
        win_size = max(3, min_side // 2 * 2 + 1)
        if min_side >= win_size:
            ssim_value = ssim(original_np, reconstructed_np, data_range=1.0, channel_axis=-1, win_size=win_size)
            ssim_loss = torch.tensor(1.0 / ssim_value, dtype=torch.float32)
        else:
            ssim_loss = torch.tensor(0.0, dtype=torch.float32)

        # 计算 LPIPS 损失
        original_lpips = original.detach().cpu()
        reconstructed_lpips = reconstructed.detach().cpu()
        lpips_value = self.lpips_loss(original_lpips, reconstructed_lpips)
        lpips_loss = lpips_value.mean()

        # 计算熵损失
        probs = F.softmax(latent, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        # 计算总损失
        total_loss = mse + 0.1*psnr_loss + 0.1*ssim_loss + lpips_loss
        print(f"mse {mse:.4f},psnr {psnr_loss:.4f},ssim {ssim_loss:.4f},lpips {lpips_loss:.4f},entropy {entropy:.4f}")
        return total_loss

'''

0.1 * mse + 0.4 * psnr_loss + 0.4 * ssim_loss + 0.1 * lpips_loss + 0.1 * entropy
1. 调整 PSNR 和 SSIM 损失权重占比，之前的图片质量占比太低，由原来的 0.01 --> 0.05
2. 使用 PSNR 和 SSIM 的倒数，而不是负数，避免修改的权重占比造成 loss 出现负值
3. 添加 LPIPS 评估，LPIPS 的值越低，图片质量越好，因此在损失函数中直接使用 LPIPS 值
4. 设置 LPIPS 的权重较大，占比 0.1，引导模型提高 LPIPS 评估值
'''
