"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np

from .window import WindowAttention, window_partition, window_reverse
import math
from timm.models.layers import DropPath, to_2tuple

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionPerformer(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., kernel_ratio=0.5):
        super().__init__()
        self.head_dim = dim // num_heads
        self.emb = dim
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(proj_drop)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.epsilon = 1e-8  # for stable in division
        self.drop_path = nn.Identity()

        self.m = int(self.head_dim * kernel_ratio)
        self.w = torch.randn(self.head_cnt, self.m, self.head_dim)
        for i in range(self.head_cnt):
            self.w[i] = nn.Parameter(nn.init.orthogonal_(self.w[i]) * math.sqrt(self.m), requires_grad=False)
        self.w.requires_grad_(False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, 1, self.m) / 2
        wtx = torch.einsum('bhti,hmi->bhtm', x.float(), self.w.to(x.device))

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attn(self, x):
        B, N, C = x.shape
        kqv = self.kqv(x).reshape(B, N, 3, self.head_cnt, self.head_dim).permute(2, 0, 3, 1, 4)
        k, q, v = kqv[0], kqv[1], kqv[2] # (B, H, T, hs)
        
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, H, T, m), (B, H, T, m)
        D = torch.einsum('bhti,bhi->bht', qp, kp.sum(dim=2)).unsqueeze(dim=-1)  # (B, H, T, m) * (B, H, m) -> (B, H, T, 1)
        kptv = torch.einsum('bhin,bhim->bhnm', v.float(), kp)  # (B, H, emb, m)
        y = torch.einsum('bhti,bhni->bhtn', qp, kptv) / (D.repeat(1, 1, 1, self.head_dim) + self.epsilon)  # (B, H, T, emb)/Diag

        # skip connection

        y = y.permute(0, 2, 1, 3).reshape(B, N, self.emb)
        y = self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def forward(self, x):
        x = self.attn(x)
        return x

class NormalCell(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, class_token=False, group=64, tokens_type='transformer', 
                shift_size=0, window_size=0, img_size=224, relative_pos=False):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.class_token = class_token
        self.img_size = img_size
        self.window_size = window_size
        self.tokens_type = tokens_type
        self.shift_size = shift_size
        if tokens_type == 'transformer':
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif tokens_type == 'performer':
            self.attn = AttentionPerformer(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif tokens_type == 'window':
            if self.shift_size > 0:
                # calculate attention mask for SW-MSA
                H, W = self.img_size, self.img_size
                img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                attn_mask = None

            self.register_buffer("attn_mask", attn_mask)
            self.attn = WindowAttention(
                in_dim=dim, out_dim=dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, relative_pos=relative_pos)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.PCM = nn.Sequential(
                            nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
                            nn.BatchNorm2d(mlp_hidden_dim),
                            nn.SiLU(inplace=True),
                            nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group),
                            nn.BatchNorm2d(dim),
                            nn.SiLU(inplace=True),
                            nn.Conv2d(dim, dim, 3, 1, 1, 1, group),
                            )

    def forward(self, x):
        b, n, c = x.shape
        shortcut = x
        if self.tokens_type == 'window':
            H, W = self.img_size, self.img_size
            assert n == self.img_size * self.img_size, "input feature has wrong size"
            x = self.norm1(x)
            x = x.view(b, H, W, c)
            padding_td = (self.window_size - H % self.window_size) % self.window_size
            padding_top = padding_td // 2
            padding_down = padding_td - padding_top
            padding_lr = (self.window_size - W % self.window_size) % self.window_size
            padding_left = padding_lr // 2
            padding_right = padding_lr - padding_left
            if padding_td + padding_lr > 0:
                x = x.permute(0, 3, 1, 2)
                x = nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_down))
                x = x.permute(0, 2, 3, 1).contiguous()

            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nW*B, window_size*window_size, C
            # W-MSA/SW-MSA
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
            shifted_x = window_reverse(attn_windows, self.window_size, H+padding_td, W+padding_lr)  # B H' W' C

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x
            x = x[:, padding_top:padding_top+H, padding_left:padding_left+W, :]
            x = x.reshape(b, H * W, c)
        else:
            x = self.attn(self.norm1(x))

        wh = int(math.sqrt(n))
        convX = self.drop_path(self.PCM(shortcut.view(b, wh, wh, c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(b, n, c))
        x = shortcut + self.drop_path(x) + convX

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x