"""SynthVision — StyleGAN2-inspired GAN for Image Generation"""
import torch, torch.nn as nn, argparse

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, layers=8):
        super().__init__()
        self.net = nn.Sequential(*[nn.Sequential(nn.Linear(z_dim if i==0 else w_dim, w_dim), nn.LeakyReLU(0.2)) for i in range(layers)])
    def forward(self, z):
        return self.net(nn.functional.normalize(z, dim=1))

class ModulatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, w_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=kernel//2)
        self.modulation = nn.Linear(w_dim, in_ch)
        self.demodulation = True
    def forward(self, x, w):
        b, c, h, ww = x.shape
        style = self.modulation(w).view(b, 1, c, 1, 1)
        weight = self.conv.weight.unsqueeze(0) * (style + 1)
        if self.demodulation:
            d = (weight.pow(2).sum(dim=[2,3,4]) + 1e-8).rsqrt()
            weight = weight * d.view(b, -1, 1, 1, 1)
        x = x.view(1, b*c, h, ww)
        weight = weight.view(b*self.conv.weight.shape[0], c, *self.conv.weight.shape[2:])
        return nn.functional.conv2d(x, weight, padding=self.conv.padding[0], groups=b).view(b, -1, h, ww)

class Discriminator(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()
        chs = [3, 64, 128, 256, 512, 512]
        layers = []
        for i in range(len(chs)-1):
            layers += [nn.Conv2d(chs[i], chs[i+1], 4, 2, 1), nn.LeakyReLU(0.2)]
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 1))
    def forward(self, x): return self.net(x)

if __name__ == "__main__":
    mapping = MappingNetwork(); disc = Discriminator()
    z = torch.randn(2, 512); w = mapping(z)
    print(f"Mapping: z{z.shape} -> w{w.shape}")
    img = torch.randn(2, 3, 256, 256); score = disc(img)
    print(f"Discriminator output: {score.shape}")
