# Networks/FFNet_S.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
# We need to import the original FFNet modules
from Networks.FFNet import Fusion, ChannelAttention, SpatialAttention, LayerNorm2d

__all__ = ['FFNet_S']

class Backbone_S(nn.Module):
    """
    MobileNetV3-Small backbone for FFNet-S.
    Extracts features from 3 different stages.
    """
    def __init__(self):
        super(Backbone_S, self).__init__()
        
        # Load pretrained MobileNetV3-Small features
        feats = list(mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features.children())
        
        # Re-defining stages to match the original FFNet's /8, /16, /32 downsampling
        self.s1 = nn.Sequential(*feats[0:4])  # 24 channels, /8
        self.s2 = nn.Sequential(*feats[4:9])  # 48 channels, /16
        self.s3 = nn.Sequential(*feats[9:12]) # 96 channels, /32
        
    def forward(self, x):
        x = x.float()
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        return s1, s2, s3

class ccsm_s(nn.Module):
    """
    Simplified FTM (ccsm) module for FFNet-S, as described in the paper.
    Uses standard convolutions for efficiency.
    """
    def __init__(self, in_channel, mid_channel, out_channel):
        super(ccsm_s, self).__init__()
        self.ch_att_s = ChannelAttention(in_channel)
        self.sa_s = SpatialAttention(7)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm2d((mid_channel,), eps=1e-06, elementwise_affine=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            LayerNorm2d((out_channel,), eps=1e-06, elementwise_affine=True)
        )

    def forward(self, x):
        x = self.ch_att_s(x) * x
        x_res = self.conv1(x) # Save residual
        x = self.conv2(x_res)
        x = self.sa_s(x) * x
        return x

class FFNet_S(nn.Module):
    def __init__(self):
        super().__init__()
        num_filters = [16, 32, 64]
        
        self.backbone = Backbone_S()
        
        # Simplified FTMs (ccsm_s) with reduced channels
        # Inputs from backbone are 24, 48, 96
        # Outputs need to match the original Fusion module's expectations
        # x1 (/8) -> num_filters[0]
        # x2 (/16) -> num_filters[1]
        # x3 (/32) -> num_filters[2]
        self.ccsm1 = ccsm_s(24, 16, num_filters[0]) # /8
        self.ccsm2 = ccsm_s(48, 24, num_filters[1]) # /16
        self.ccsm3 = ccsm_s(96, 48, num_filters[2]) # /32

        # Original Fusion module from FFNet.py
        # It handles the upsampling internally.
        self.fusion = Fusion(num_filters[0], num_filters[1], num_filters[2])

    def forward(self, x):
        s1, s2, s3 = self.backbone(x)
        
        f1 = self.ccsm1(s1)
        f2 = self.ccsm2(s2)
        f3 = self.ccsm3(s3)

        x = self.fusion(f1, f2, f3)
 
        # Normalize density map (from original FFNet)
        B, C, H, W = x.size()
        x_sum = x.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x_normed = x / (x_sum + 1e-6)

        return x, x_normed

if __name__ == '__main__':
    x = torch.rand(size=(4, 3, 256, 256), dtype=torch.float32)
    model_s = FFNet_S()
    
    # Test model
    mu, mu_norm = model_s(x)
    print(f"FFNet-S Output size: {mu.size()}, {mu_norm.size()}")
    
    total_params = sum(p.numel() for p in model_s.parameters() if p.requires_grad)
    print(f"FFNet-S Total Parameters: {total_params / 1e6:.2f}M") # Should be ~3.1M