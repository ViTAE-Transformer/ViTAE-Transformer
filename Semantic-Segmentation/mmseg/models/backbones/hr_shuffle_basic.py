import torch
from torch import nn, einsum
from torch.jit import Error
from einops import rearrange, repeat
import numpy as np
import math
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from ..builder import BACKBONES

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0., stride=False):
        super().__init__()
        self.stride = stride
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, window_size=1, shuffle=False, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., relative_pos_embedding=False):
        super().__init__()
        self.num_heads = num_heads
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        self.shuffle = shuffle

        self.scale = qk_scale or head_dim ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

    def forward(self, x, window_mask=None):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)

        if self.shuffle:
            if window_mask is not None:
                window_mask = window_mask.reshape(b, self.ws, h // self.ws, self.ws, w // self.ws).permute(0, 2, 4, 1, 3).reshape(-1, self.ws * self.ws)
                attn_mask = window_mask[:, None, :] + window_mask[:, :, None]
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-200.0)).masked_fill(attn_mask == 0, float(0.0))
            q, k, v = rearrange(qkv, 'b (qkv h d) (ws1 hh) (ws2 ww) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)
        else:
            if window_mask is not None:
                window_mask = window_mask.reshape(b, h // self.ws, self.ws, w // self.ws, self.ws).permute(0, 1, 3, 2, 4).reshape(-1, self.ws * self.ws)
                attn_mask = window_mask[:, None, :] + window_mask[:, :, None]
                attn_mask = attn_mask.masked_fill(attn_mask > 0.1, float(-200.0)).masked_fill(attn_mask < 0.1, float(0.0))
            q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        # print(attn_mask.shape)
        # print(dots.shape)
        if window_mask is not None:
            dots += attn_mask.unsqueeze(1)

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        if self.shuffle:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (ws1 hh) (ws2 ww)', h=self.num_heads, b=b, hh=h//self.ws, ws1=self.ws, ws2=self.ws)
        else:
            out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=h//self.ws, ws1=self.ws, ws2=self.ws)
 
        out = self.proj(out)
        out = self.proj_drop(out)

        return out

class Block(nn.Module):
    def __init__(self, dim, out_dim, num_heads, window_size=1, shuffle=False, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, stride=False, relative_pos_embedding=False):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, relative_pos_embedding=relative_pos_embedding)
        self.local = nn.Conv2d(dim, dim, window_size, 1, window_size//2, groups=dim, bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop, stride=stride)
        self.norm3 = norm_layer(dim)
        print("input dim={}, output dim={}, stride={}, expand={}, num_heads={}".format(dim, out_dim, stride, shuffle, num_heads))

    def forward(self, x):
        B, _, H, W = x.shape
        # print(H, W)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))
        # print(x.shape)
        window_mask = torch.zeros(B, H+pad_b, W+pad_r).to(x.device).fill_(1.0)
        window_mask[:, :H, :W] = 0

        # _, Hp, Wp, _ = x.shape

        x = x + self.drop_path(self.attn(self.norm1(x), window_mask))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        x = x + self.local(self.norm2(x)) # local connection
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.norm = norm_layer(dim)
        self.reduction = nn.Conv2d(dim, out_dim, 2, 2, 0, bias=False)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f"input dim={self.dim}, out dim={self.out_dim}"


class StageModule(nn.Module):
    def __init__(self, layers, dim, out_dim, num_heads, window_size=1, shuffle=True, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, relative_pos_embedding=False):
        super().__init__()
        # assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        if dim != out_dim:
            self.patch_partition = PatchMerging(dim, out_dim)
        else:
            self.patch_partition = None

        num = layers // 2
        self.layers = nn.ModuleList([])
        for idx in range(num):
            the_last = (idx==num-1)
            self.layers.append(nn.ModuleList([
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=False, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                      relative_pos_embedding=relative_pos_embedding),
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=shuffle, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
                      relative_pos_embedding=relative_pos_embedding)
            ]))
        if layers % 2 == 1:
            self.layers.append(nn.ModuleList([
                Block(dim=out_dim, out_dim=out_dim, num_heads=num_heads, window_size=window_size, shuffle=False, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                      relative_pos_embedding=relative_pos_embedding),
                nn.Identity()
            ]))
        self.num = num

    def forward(self, x):
        if self.patch_partition:
            x = self.patch_partition(x)
            
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
            
        return x

class FusionBlock(nn.Module):
    def __init__(self, token_dims, window_size, levels_num, cur_level, cross_attn_type,
                act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, ):
        super().__init__()
        self.levels_num = levels_num
        self.cur_level = cur_level
        self.cross_attn_type = cross_attn_type
        self.downsampling_layer = nn.ModuleList([])
        # print(cur_level)
        for i in range(0, cur_level):
            # print(2**(cur_level-i))
            self.downsampling_layer.append(
                Downsampling(ratio=2**(cur_level-i), type='linear', in_chans=token_dims[i], out_chans=token_dims[cur_level], norm_layer=norm_layer, act_layer=act_layer)
            )
        for i in range(cur_level+1, levels_num):
            self.downsampling_layer.append(
                ToDim(kernel_size=1, stride=1, in_chans=token_dims[i], out_chans=token_dims[cur_level], norm_layer=norm_layer, act_layer=act_layer)
            )
        if cross_attn_type == 'convcat':
            self.conv1 = nn.Conv2d(token_dims[self.cur_level]*self.levels_num, token_dims[self.cur_level], 1, 1)

        # ratio = 1
        # self.ratios = []
        # downsample_ratios.reverse()
        # for i in range(levels_num-1):
        #     ratio *= downsample_ratios[i]
        #     self.ratios.append(ratio)
        # self.ratios.insert(0, 1)
        # # self.ratios = ([1].extend(ratios)).reverse()
        # # downsample_ratios = 
        # self.token_dims = token_dims
        # self.levels_num = levels_num
        # self.cur_level = cur_level
        # self.window_size = window_size
        # self.cross_attn_type = cross_attn_type
        # self.downsampling_layer = nn.ModuleList([])
        # for i in range(0, self.cur_level):
        #     ratio = self.ratios[self.cur_level] // self.ratios[i]
        #     self.downsampling_layer.append(
        #         Downsampling(ratio=ratio, type=cross_downsampling_type, in_chans=token_dims[i], out_chans=token_dims[cur_level], norm_layer=norm_layer, act_layer=act_layer)
        #     )
        # for i in range(self.cur_level+1, self.levels_num):
        #     self.downsampling_layer.append(
        #         ToDim(kernel_size=1, stride=1, in_chans=token_dims[i], out_chans=token_dims[cur_level], norm_layer=norm_layer, act_layer=act_layer)
        #     )
        
        # # self.fusion = nn.ModuleList([])
        # # for i in range(self.cur_level, self.levels_num):
        # dim = token_dims[self.cur_level]
        # self.norm1 = norm_layer(dim)
        # if cross_attn_type == 'hybrid':
        #     # self.attn = HybridCrossAttn(dim=token_dims[self.cur_level], num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, window_size=self.window_size, levels_num=levels_num)
        #     # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()       #TODO: check whether using drop path for cross attn is necessary
        #     # self.norm2 = norm_layer(dim)
        #     # mlp_hidden_dim = int(dim * mlp_ratio)
        #     # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
        #     pass
        # elif cross_attn_type == 'hybridconcat':
        #     pass
        #     # self.fusion = ConvAddFusion()
        # elif cross_attn_type == 'convadd':
        #     pass
        # elif cross_attn_type == 'convcat':
        #     self.conv1 = nn.Conv2d(token_dims[self.cur_level]*self.levels_num, token_dims[self.cur_level], 1, 1)
        # else:
        #     raise NotImplementedError(f'not implemented for {cross_attn_type} cross attn yet!')
    
    def forward(self, x):
        assert len(x) == self.levels_num
        shapes = []
        for i in range(self.levels_num):
            shapes.append(x[i].shape)
        B, C, H, W = x[self.cur_level].shape
        _x = []
        for i in range(0, self.cur_level):
            _x.append(self.downsampling_layer[i](x[i]))
            # _x.append(
            #     window_partition(self.downsampling_layer[i](x[i]).permute(0,2,3,1), self.window_size, spatial_shuffle=False)
            # )
        for i in range(self.cur_level+1, self.levels_num):   # no current level tokens
            # B, _, H, W = x[i].shape
            _x.append(self.downsampling_layer[i-1](x[i]))
            # x_tmp = window_partition(self.downsampling_layer[i-1](x[i]).permute(0,2,3,1), self.window_size, spatial_shuffle=False)   #B*N, window_size ** 2, C
            # x_tmp = x_tmp.unsqueeze(dim=1).repeat(1, (self.ratios[i] // self.ratios[self.cur_level])**2, 1, 1).reshape(-1, self.window_size**2, C)
            # _x.append(x_tmp)
        # _x = torch.cat(_x, dim=1)
        # _x = window_partition(_x.permute(0,2,3,4), self.window_size, spatial_shuffle=False)
        x = x[self.cur_level]

        # x = window_partition(x.permute(0,2,3,1), self.window_size, spatial_shuffle=False)
        if self.cross_attn_type == 'hybrid':
            # for i in range(0, self.cur_level):   # no current level tokens
            #     _x[i] = window_partition( _x[i].permute(0,2,3,1), self.window_size, spatial_shuffle=False)   #B*N, window_size ** 2, C
            # for i in range(self.cur_level+1, self.levels_num):   # no current level tokens
            #     _x[i] = window_partition( _x[i].permute(0,2,3,1), self.window_size, spatial_shuffle=False)   #B*N, window_size ** 2, C
            #     _x[i] = _x[i].unsqueeze(dim=1).repeat(1, (self.ratios[i] // self.ratios[self.cur_level])**2, 1, 1).reshape(-1, self.window_size**2, C)
            # x = x + self.drop_path(self.attn(self.norm1(x), _x))       #TODO: check drop path
            # x = x + self.drop_path(self.mlp(self.norm2(x)))
            raise Error('no such type')
        elif self.cross_attn_type == 'convadd':
            for i in range(len(_x)):
                x += F.interpolate(_x[i], size=x.shape[-2:], mode='bilinear', align_corners=True)
        elif self.cross_attn_type == 'convcat':
            # out = []
            for i in range(len(_x)):
                x = torch.cat([(F.interpolate(_x[i], size=x.shape[-2:], mode='bilinear', align_corners=True)), x], dim=1)
            x = self.conv1(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU6(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        return x

class ToDim(nn.Module):
    def __init__(self, kernel_size, stride, in_chans, out_chans, norm_layer, act_layer, bias=True, padding=0):
        '''
        type='avg' avgpooling; 'max' maxpooling; 'unmax' maxunpooling; 'linear' linear transform;
        '''
        if isinstance(norm_layer(1), nn.LayerNorm):
            norm_layer = nn.BatchNorm2d
        super().__init__()
        self.layer = nn.Sequential(*[nn.Conv2d(in_channels=in_chans, out_channels=out_chans, kernel_size=kernel_size, stride=stride, bias=False, padding=padding), norm_layer(out_chans), act_layer()])
    
    def forward(self, x):
        return self.layer(x)

class Downsampling(nn.Module):
    '''
    args:
        type='avg' avgpooling; 'max' maxpooling; 'unmax' maxunpooling; 'linear' linear transform;
    '''
    def __init__(self, ratio, type, in_chans, out_chans, norm_layer, act_layer):
        
        if isinstance(norm_layer(1), nn.LayerNorm):
            norm_layer = nn.BatchNorm2d
        super().__init__()
        if type=='avg':
            self.downsampling_layer = nn.Sequential(
                *[nn.AvgPool2d(kernel_size=ratio, stride=ratio), ToDim(kernel_size=1, stride=1, in_chans=in_chans, out_chans=out_chans, norm_layer=norm_layer, act_layer=act_layer)]
            )
        elif type == 'max':
            self.downsampling_layer = nn.Sequential(
                *[nn.MaxPool2d(kernel_size=ratio, stride=ratio), ToDim(kernel_size=1, stride=1, in_chans=in_chans, out_chans=out_chans, norm_layer=norm_layer, act_layer=act_layer)]
            )
        elif type == 'unmax':
            self.downsampling_layer = nn.Sequential(
                *[nn.MaxUnpool2d(kernel_size=ratio, stride=ratio), ToDim(kernel_size=1, stride=1, in_chans=in_chans, out_chans=out_chans, norm_layer=norm_layer, act_layer=act_layer)]
            )
        elif type == 'linear':
            self.downsampling_layer = []
            for i in range(int(math.log2(ratio))):
                self.downsampling_layer.append(
                    ToDim(kernel_size=2, stride=2, in_chans=in_chans, out_chans=in_chans*2, norm_layer=norm_layer, act_layer=act_layer)
                    # nn.Sequential(*[nn.Conv2d(in_channels=in_chans, out_channels=in_chans*2, kernel_size=2, stride=2), norm_layer(in_chans*2), act_layer()])
                )
                in_chans = in_chans*2
            # in_chans = in_chans // 2
            if in_chans != out_chans:
                self.downsampling_layer[-1] = ToDim(kernel_size=2, stride=2, in_chans=in_chans // 2, out_chans=out_chans, norm_layer=norm_layer, act_layer=act_layer)
            self.downsampling_layer = nn.Sequential(*self.downsampling_layer)
            # assert in_chans == out_chans
        else:
            raise NotImplementedError('No such downsampling type!')
    
    def forward(self, x):
        return self.downsampling_layer(x)

class GenerateNewBranch(nn.Module):
    '''
    args:
        token_dims: token dims for each level
        levels_num: num of levels currently
        norm_layer
        act_layer
    '''
    def __init__(self, token_dims, levels_num, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6):
        super().__init__()
        # ratio = 1
        # self.ratios = []
        # downsample_ratios.reverse()
        # for i in range(levels_num):
        #     ratio *= downsample_ratios[i]
        #     self.ratios.append(ratio)
        # self.ratios.insert(0, 1)
        # self.ratios.reverse()
        self.token_dims = token_dims
        self.levels_num = levels_num
        # self.cur_level = cur_level
        self.downsampling_layer = nn.ModuleList([])
        for i in range(self.levels_num):
            ratio = 2**(self.levels_num-i)
            self.downsampling_layer.append(
                Downsampling(ratio=ratio, type='linear', in_chans=token_dims[i], out_chans=token_dims[levels_num], norm_layer=norm_layer, act_layer=act_layer)
            )
        # self.conv1 = ToDim(kernel_size=1, stride=1, in_chans=token_dims[levels_num] * levels_num, out_chans=token_dims[levels_num], norm_layer=nn.BatchNorm2d, act_layer=act_layer)
        self.conv1 = nn.Sequential(*[nn.Conv2d(token_dims[levels_num] * levels_num, token_dims[levels_num], 1, 1)])
        
    def forward(self, x):
        assert self.levels_num == len(x)
        _x = []
        for i in range(len(x)):
            _x.append(
                self.downsampling_layer[i](x[i])
            )
        x = torch.cat(_x, dim=1)
        x = self.conv1(x)
        return x

@BACKBONES.register_module()
class HrShuffleBasic(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, token_dim=32, embed_dim=96, mlp_ratio=4., depths=[2,2,2,2], layer_ratio=[1,2,3,4], num_heads=[3,6,12,24], 
                relative_pos_embedding=True, shuffle=True, window_size=7, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                has_pos_embed=False, out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.has_pos_embed = has_pos_embed
        dims = [i*32 for i in num_heads]
        assert dims[0] == embed_dim
        self.dims = dims
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.to_token = PatchEmbedding(inter_channel=token_dim, out_channels=embed_dim)

        num_patches = (img_size*img_size) // 16

        if self.has_pos_embed:
            self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches, d_hid=embed_dim), requires_grad=False)
            self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]  # stochastic depth decay rule
        self.branch = nn.ModuleList([])
        for i in range(4):
            _layer = nn.ModuleList([])
            for j in range(0, i+1):
                _layer.append(
                    StageModule(depths[i]*layer_ratio[j], dims[j], dims[j], num_heads[j], window_size=window_size, shuffle=shuffle,
                                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                                  relative_pos_embedding=relative_pos_embedding)
                )
            self.branch.append(_layer)
        
        self.generate_new_branch = nn.ModuleList([])
        for i in range(3):
            self.generate_new_branch.append(
                GenerateNewBranch(token_dims=dims, levels_num=i+1, )
            )
        
        self.fusion = nn.ModuleList([])
        self.fusion.append(nn.Identity())
        
        for i in range(1, 4):
            _fusion = nn.ModuleList([])
            for j in range(i+1):
                _fusion.append(
                    FusionBlock(dims, window_size, levels_num=i+1, cur_level=j, cross_attn_type='convcat', )
                )
            self.fusion.append(_fusion)

        norms = []
        for i in range(4):
            norms.append(nn.BatchNorm2d(self.dims[i]))
        self.norms = nn.ModuleList(norms)
        # self.stage1 = StageModule(layers[0], embed_dim, dims[0], num_heads[0], window_size=window_size, shuffle=shuffle,
        #                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0],
        #                           relative_pos_embedding=relative_pos_embedding)
        # self.stage2 = StageModule(layers[1], dims[0], dims[1], num_heads[1], window_size=window_size, shuffle=shuffle,
        #                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1],
        #                           relative_pos_embedding=relative_pos_embedding)
        # self.stage3 = StageModule(layers[2], dims[1], dims[2], num_heads[2], window_size=window_size, shuffle=shuffle,
        #                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[2],
        #                           relative_pos_embedding=relative_pos_embedding)
        # self.stage4 = StageModule(layers[3], dims[2], dims[3], num_heads[3], window_size=window_size, shuffle=shuffle, 
        #                           mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[3],
        #                           relative_pos_embedding=relative_pos_embedding)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Classifier head
        # self.head = nn.Linear(np.sum(dims), num_classes) if num_classes > 0 else nn.Identity()
        # self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)
        
        
        self._freeze_stages()

        # Classifier head
        # self.head = nn.Linear(self.tokens_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)
    def _freeze_stages(self):

        if self.frozen_stages > 0:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    # def _init_weights(self, m):
    #     if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
    #         nn.init.constant_(m.weight, 1.0)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, (nn.Linear, nn.Conv2d)):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.to_token(x)
        b, c, h, w = x.shape

        if self.has_pos_embed:
            x = x + self.pos_embed.view(1, h, w, c).permute(0, 3, 1, 2)
            x = self.pos_drop(x)

        x = [x]
        for i in range(4):
            assert len(x) == i+1
            for j in range(i+1):
                x[j] = self.branch[i][j](x[j])
            x_next = []
            if i != 0:
                for j in range(i+1):
                    x_next.append(
                        self.fusion[i][j](x)
                    )
            else:
                x_next.extend(x)
            if i != 3:
                x_next.append(
                    self.generate_new_branch[i](x)
                )
            x = x_next

        # x = self.stage1(x)
        # x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        # for i in range(len(x)):
        #     x[i] = self.avgpool(x[i])

        # x = torch.cat([self.avgpool(y) for y in x], dim=1)

        # x = self.avgpool(x)
        # x = self.avgpool(x[-1])
        # x = torch.flatten(x, 1)
        for i in range(len(x)):
            x[i] = self.norms[i](x[i])
        return tuple(x)

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #     return x
    
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(HrShuffleBasic, self).train(mode)
        self._freeze_stages()