import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum

import torch.fft


class GaussianImage_RS(nn.Module):
    def __init__(self, dim=192, num_points=4, size_HW=4, size_BLOCK_HW=2):
        super().__init__()

        self.init_num_points = num_points
        self.H, self.W = size_HW, size_HW
        self.BLOCK_W, self.BLOCK_H = size_BLOCK_HW, size_BLOCK_HW
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H, 1,) # 
        
        self.device = "cuda"
        # self.device = "cpu"

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._scaling = nn.Parameter(torch.rand(self.init_num_points, 2))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._rotation = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))

        self.last_size = (self.H, self.W)
        self.background = torch.ones(3, device=self.device)
        self.rotation_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))

        # FFT
        self.proj_up = nn.Linear(3, dim)

    @property
    def get_scaling(self):
        return torch.abs(self._scaling+self.bound)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)*2*math.pi
    
    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity 
    
    def forward(self, x):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(self.get_xyz, self.get_scaling, self.get_rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
    
        out_img = torch.clamp(out_img, 0, 1).reshape(-1, self.H*self.W, 3) 
        x_out = self.proj_up(out_img).permute(0,2,1).reshape(-1,x.shape[1],self.H,self.W)
        
        return x_out


class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=4, w=3):
        super().__init__()
    
        hidden_dim = dim * 2
        self.conv_in = nn.Conv2d(dim, hidden_dim*2, kernel_size=1)
        self.complex_weight = nn.Parameter(torch.randn(h, w, hidden_dim*2, 2, dtype=torch.float32) * 0.02)
        self.conv_middle = nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        batch, a, b, C = x.shape

        x = x.to(torch.float32)
        x = self.conv_in(self.norm(x).permute(0,3,1,2)).permute(0,2,3,1)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x1, x2 = self.conv_middle(x.permute(0,3,1,2)).chunk(2, dim=1)
        x = F.gelu(x1) * x2

        return x


class TokenGeneration(nn.Module):
    def __init__(self, dim, topk_win_num=3):
        super().__init__()
        
        self.GS_s = GaussianImage_RS(dim=dim, num_points=9, size_HW=4, size_BLOCK_HW=2)
        self.max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.topk_win_num = topk_win_num
        self.winSize = 2

        # FFT
        self.filter = SpectralGatingNetwork(dim)

    def forward(self, z, x):
        # template
        B, N_t, C = z.shape
        h_t = int(math.sqrt(N_t))
        z_center = z.permute(0,2,1).reshape(B,C,h_t,h_t)
        z_max = self.max_pool(z_center).permute(0,2,3,1).reshape(B,1,C)

        # search region
        N_s = x.shape[1]
        h_s = int(math.sqrt(N_s))
        win_Size_all = int(self.winSize*self.winSize)
        win_Num_H = h_s//self.winSize
        
        sim_x = ((F.normalize(z_max,dim=-1,p=2) @ F.normalize(x,dim=-1,p=2).transpose(-2,-1))).reshape(B,N_s)
        sim_x = sim_x.reshape(B,win_Num_H,self.winSize,win_Num_H,self.winSize).permute(0,1,3,2,4)
        sim_x = (sim_x.reshape(B,-1,win_Size_all)).mean(dim=-1)
        index_x_T = torch.topk(sim_x,k=self.topk_win_num,dim=-1)[1] # [B,win_topk]
        index_x_T = index_x_T.unsqueeze(dim=-1).unsqueeze(dim=-1).expand(-1,-1,win_Size_all,C)

        x_ext = x.reshape(B,win_Num_H,self.winSize,win_Num_H,self.winSize,C)
        x_ext = x_ext.permute(0,1,3,2,4,5).reshape(B,-1,win_Size_all,C)
        x_ext = torch.gather(x_ext,dim=1,index=index_x_T)
        x_ext = x_ext.permute(0,1,3,2).reshape(B*self.topk_win_num,C,self.winSize,self.winSize)
        x_ext = F.interpolate(x_ext, scale_factor=2, mode='bilinear', align_corners=False)

        # FFT
        x_ext = self.filter(x_ext.permute(0,2,3,1)) + x_ext

        # GS
        x_ext = (self.GS_s(x_ext)+x_ext).reshape(B,self.topk_win_num,C,-1)
        x_ext = x_ext.permute(0,1,3,2).reshape(B,-1,C)

        return x_ext