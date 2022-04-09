"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath
import math

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
        self.emb = dim * num_heads # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(proj_drop)
        self.proj = nn.Linear(self.emb, self.emb)
        self.head_cnt = num_heads
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.epsilon = 1e-8  # for stable in division
        self.drop_path = nn.Identity()

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch 
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def forward(self, x):
        x = self.attn(x)
        return x

class NormalCell(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, class_token=False, group=64, tokens_type='transformer'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.class_token = class_token
        if 'transformer' in tokens_type:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif 'performer' in tokens_type:
            self.attn = AttentionPerformer(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if 'shallow' in tokens_type:
            self.PCM = nn.Sequential(
                            nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
                            nn.BatchNorm2d(mlp_hidden_dim),
                            nn.SiLU(inplace=True),
                            nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group)
                            )
        else:
            self.PCM = nn.Sequential(
                                nn.Conv2d(dim, mlp_hidden_dim, 3, 1, 1, 1, group),
                                nn.BatchNorm2d(mlp_hidden_dim),
                                nn.SiLU(inplace=True),
                                nn.Conv2d(mlp_hidden_dim, dim, 3, 1, 1, 1, group),
                                nn.BatchNorm2d(dim),
                                nn.SiLU(inplace=True),
                                nn.Conv2d(dim, dim, 3, 1, 1, 1, group),
                                nn.SiLU(inplace=True),
                                )

    def forward(self, x):
        b, n, c = x.shape
        if self.class_token:
            n = n - 1
            wh = int(math.sqrt(n))
            convX = self.drop_path(self.PCM(x[:, 1:, :].view(b, wh, wh, c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(b, n, c))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x[:, 1:] = x[:, 1:] + convX
        else:
            wh = int(math.sqrt(n))
            convX = self.drop_path(self.PCM(x.view(b, wh, wh, c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(b, n, c))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + convX
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
