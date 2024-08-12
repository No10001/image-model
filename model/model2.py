import torch
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(self, num_channels):
        super(GDN, self).__init__()
        self.gamma = nn.Parameter(torch.sqrt(torch.ones(1, num_channels, 1, 1)))
        self.eps = 1e-6

    def forward(self, x):
        squared_sum = (x.pow(2)).sum(dim=1, keepdim=True)
        norm = torch.sqrt(squared_sum + self.eps)
        return x / (norm * self.gamma)

class IGDN(nn.Module):
    def __init__(self, num_channels):
        super(IGDN, self).__init__()
        self.gamma = nn.Parameter(torch.sqrt(torch.ones(1, num_channels, 1, 1)))
        self.eps = 1e-6

    def forward(self, x):
        return x * (self.gamma + self.eps)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(16)
        self.gdn0 = GDN(16)
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.gdn1 = GDN(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.gdn2 = GDN(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.gdn3 = GDN(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.gdn4 = GDN(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.gdn5 = GDN(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.gdn6 = GDN(1024)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.gdn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.gdn4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.gdn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.gdn6(x)
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconv6 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.upconv0 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn0 = nn.BatchNorm2d(8)
        self.conv = nn.ConvTranspose2d(8, 3, kernel_size=3, stride=1, padding=1)
        self.igdn = IGDN(8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn6(self.upconv6(x)))
        x = F.relu(self.bn5(self.upconv5(x)))
        x = F.relu(self.bn4(self.upconv4(x)))
        x = F.relu(self.bn3(self.upconv3(x)))
        x = F.relu(self.bn2(self.upconv2(x)))
        x = F.relu(self.bn1(self.upconv1(x)))
        x = self.igdn(self.bn0(self.upconv0(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.conv(x))
        return x

class ImageAutoencoder(nn.Module):
    def __init__(self):
        super(ImageAutoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class HyperpriorAutoencoder(nn.Module):
    def __init__(self, N, M):
        super(HyperpriorAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, M, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.Conv2d(M, M, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(M),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(M, M, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.ConvTranspose2d(M, M, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.ConvTranspose2d(M, M, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.ConvTranspose2d(M, M, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.ConvTranspose2d(M, M, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.ConvTranspose2d(M, M, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(M),
            nn.ReLU(),
            nn.ConvTranspose2d(M, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, y):
        z = self.encoder(y)
        y_reconstructed = self.decoder(z)
        return y_reconstructed, z

class ScaleHyperprior(nn.Module):
    def __init__(self, N, M):
        super(ScaleHyperprior, self).__init__()
        self.ga = ImageAutoencoder()
        self.ha = HyperpriorAutoencoder(N, M)

    def forward(self, x):
        y = self.ga(x)
        y_reconstructed, z = self.ha(y)
        return y_reconstructed, z
