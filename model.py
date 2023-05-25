# model.py
import torch
import torch.nn as nn

# Define a simple ConvBlock for ease
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

# A very simplified Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64, 7, 1, 3),  # input is (nc) x 256 x 256
            ConvBlock(64, 128, 3, 2, 1),  # state size: (ndf) x 128 x 128
            ConvBlock(128, 256, 3, 2, 1),  # state size: (ndf*2) x 64 x 64
            ConvBlock(256, 512, 3, 2, 1),  # state size: (ndf*4) x 32 x 32
            ConvBlock(512, 512, 3, 2, 1),  # state size: (ndf*4) x 16 x 16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # state size: (ndf*2) x 32 x 32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # state size: (ndf) x 64 x 64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # state size: (nc) x 128 x 128
            nn.ConvTranspose2d(64, 3, 4, 2, 1),  # state size: (nc) x 256 x 256
            nn.Tanh()
        )

    def forward(self, input):
        return self.model(input)

# A very simplified Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64, 4, 2, 1),  # input is (nc) x 256 x 256
            ConvBlock(64, 128, 4, 2, 1),  # state size: (ndf) x 128 x 128
            ConvBlock(128, 256, 4, 2, 1),  # state size: (ndf*2) x 64 x 64
            ConvBlock(256, 512, 4, 2, 1),  # state size: (ndf*4) x 32 x 32
            nn.Conv2d(512, 1, 4, 1, 1),  # state size: (ndf*4) x 31 x 31
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)
