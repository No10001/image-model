import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lpips_loss = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, original, reconstructed, latent):
        # 使用自适应池化确保 reconstructed 和 original 的尺寸匹配
        reconstructed = F.adaptive_avg_pool2d(reconstructed, original.shape[2:])

        # MSE Loss
        mse = self.mse_loss(reconstructed, original)

        # PSNR Loss
        # original_np = original.detach().numpy() #.cpu()
        original_np = original.detach().cpu().numpy() #.cpu()
        # reconstructed_np = reconstructed.detach().numpy() #.cpu()
        reconstructed_np = reconstructed.detach().cpu().numpy()
        # print('0000000000000000000000000')
        psnr_value = psnr(original_np, reconstructed_np, data_range=1.0)
        # print('222222222222222222222222222222')
        psnr_loss = torch.tensor(1.0 / psnr_value, dtype=torch.float32, device=original.device)

        # SSIM Loss
        min_side = min(original.size(2), original.size(3))
        win_size = max(3, min_side // 2 * 2 + 1)
        if min_side >= win_size:
            ssim_value = ssim(original_np, reconstructed_np, data_range=1.0, channel_axis=-1, win_size=win_size)
            ssim_loss = torch.tensor(1.0 / ssim_value, dtype=torch.float32, device=original.device)
        else:
            ssim_loss = torch.tensor(0.0, dtype=torch.float32, device=original.device)

        # LPIPS Loss
        original_lpips = original.detach()#.cpu()
        reconstructed_lpips = reconstructed.detach()#.cpu()
        lpips_value = self.lpips_loss(original_lpips, reconstructed_lpips)
        lpips_loss = lpips_value.mean()

        # 计算熵损失
        probs = F.softmax(latent, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1).mean()

        # 计算总损失
        total_loss = 0.4 * mse + 0.3 * psnr_loss + 0.3 * ssim_loss + 0.4 * lpips_loss

        return total_loss