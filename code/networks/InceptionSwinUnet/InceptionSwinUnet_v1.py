'''
         concat
          ^^^
Upsample           Upsample 
    ↑       ↑         ↑
Linear    Conv    Attention
    ↑       ↑         ↑
Maxpool   Linear   Avgpool
    ↑       ↑         ↑
X_h1      X_h2       X_l
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import ml_collections


def window_partition(x, window_size):
    '''
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    '''
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    '''
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    '''
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    '''
    MLP(x) = drop(fc2(drop(act(fc1(x)))))
    '''
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
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


class WindowAttention(nn.Module):
    '''
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    
    Returns:
        proj_drop(proj(attn_drop(softmax((q * scale) @ k^T + relative_position_bias + mask)) @ v))
    '''

    def __init__(
        self, dim, window_size, num_heads,
        qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02) # truncated normal initialize relative_position_bias_table
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0, let negtive be positive
        relative_coords[:, :, 1] += self.window_size[1] - 1  # shift to start from 0
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask=None):
        '''
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Mh*Mw, Mh*Mw) or None

        Returns:
            proj_drop(proj(attn_drop(softmax((q * scale) @ k^T + relative_position_bias + mask)) @ v))
        '''
        # [batch_size*num_windows, Mh*Mw(window_size_h, window_size_w), total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, nH, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, nH, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, nH, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, nH, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, nH, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # view(-1): [Mh*Mw, Mh*Mw] -> [Mh*Mw*Mh*Mw]
        # table[]: [Mh*Mw*Mh*Mw] -> [Mh*Mw*Mh*Mw, nH]
        # view(,,-1): [Mh*Mw*Mh*Mw, nH] -> [Mh*Mw, Mh*Mw, nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0) # [batch_size*num_windows, nH, Mh*Mw, Mh*Mw]

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            # add mask: same region + 0, different regions + (-100) (a big negtive number). After softmax, different regions will be 0.
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N) # [batch_size * num_windows, num_heads, Mh*Mw, Mh*Mw]
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply: [batch_size * num_windows, nH, Mh*Mw, Mh*Mw] * 
        #              [batch_size * num_windows, nH, Mh*Mw, embed_dim_per_head]
        #           -> [batch_size * num_windows, nH, Mh*Mw, embed_dim_per_head]
        # transpose:-> [batch_size * num_windows, Mh*Mw, nH, embed_dim_per_head]
        # reshape:  -> [batch_size * num_windows, Mh*Mw, embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    '''
    [LN →  W-MSA → LN → MLP] or [LN → SW-MSA → LN → MLP]

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    Returns:
        x: [B, H*W, C]
    '''

    def __init__(
        self, dim, input_resolution, num_heads, window_size=4, shift_size=0.,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
        act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        # because the avgpool before let the input_resolution become the half of itself
        self.input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # print('self.input_resolution', self.input_resolution)
        # print('self.window_size', self.window_size)

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
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

            mask_windows = window_partition(img_mask, self.window_size) # [nW, Mh, Mw, 1]
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)# [nW, Mh*Mw]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1] = [nW, Mh*Mw, Mh*Mw]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # print('SwinTransformerBlock', x.shape)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size {L} != {H} * {W}"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # remove padding data
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinTransformer(nn.Module):
    '''
    avgpool(X_l) -> LN →  W-MSA → LN → MLP → LN → SW-MSA → LN → MLP → upsample

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    Returns:
        x: [B, C, H*W] avgpool-> [B, C, H/2, W/2] attn-> [B, H/2*W/2, C] upsample-> [B, C, H, W]
    '''

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False
    ):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(2)
        ])

    def forward(self, x):
        # print('SwinTransformer', x.shape)
        H, W = self.input_resolution
        B, C, L = x.shape
        x = x.view(-1, C, H, W)
        x = self.avgpool(x) # [B, C, H/2, W/2]
        x = x.view(-1, C, H//2*W//2).permute(0, 2, 1)
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x) # [B, H/2*W/2, C]
        x = x.view(-1, H//2, W//2, C).permute(0, 3, 1, 2)
        x = self.upsample(x) # [B, C, H, W]
        return x


class MaxPoolLinear(nn.Module):
    '''
    y1 = upsample(fc(maxpool(X_h1)))
    '''
    
    def __init__(self, i_ch, o_ch):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(i_ch, o_ch)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.maxpool(x) # [B, C, H/2, W/2]
        x = x.permute(0, 2, 3, 1) # [B, H/2, W/2, C]
        x = self.fc(x) # [B, H/2, W/2, C]
        x = x.permute(0, 3, 1, 2) # [B, C, H/2, W/2]
        x = self.upsample(x) # [B, C, H, W]
        return x


class LinearDWConv(nn.Module):
    '''
    y2 = conv(fc(X_h2))
    '''
    def __init__(self, i_ch, o_ch):
        super().__init__()
        self.fc = nn.Linear(i_ch, i_ch)
        self.conv = nn.Conv2d(i_ch, o_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # [B, H, W, C]
        x = self.fc(x) # [B, H, W, C]
        x = x.permute(0, 3, 1, 2) # [B, C, H, W]
        x = self.conv(x) # [B, C, H, W]
        return x


class InceptionMixer(nn.Module):
    '''
    Mix the low-frequency procession and the high-frequency procession [B, HW, C] -> [B, HW, C]

    Args:
        dim (int): Number of input channels.
        alpha (float): Low frequency / high frequency.
        input_resolution (tuple[int]): Input resulotion.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.

    Returns:
        X_h1, X_h2, X_l = x.split(dim=c)
        y1 = fc(maxpool(X_h1))
        y2 = conv(fc(X_h2))
        y3 = upsample(W-MSA + SW-MSA(avepool(X_l)))
        yc = concat(y1, y2, y3)
        y  = fc(yc + conv(yc))
    '''
    def __init__(
        self, dim, alpha, input_resolution, depth, num_heads, window_size,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False
    ):
        super().__init__()
        # parameters
        self.ch_l, self.ch_h = int(dim * alpha), int(dim * (1 - alpha))
        self.ch_h1, self.ch_h2 = self.ch_h // 2, self.ch_h // 2
        self.input_resolution = input_resolution

        # operations
        self.swintransformer = SwinTransformer(
            dim=self.ch_l, input_resolution=input_resolution, depth=depth, num_heads=num_heads, window_size=window_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, downsample=downsample, use_checkpoint=use_checkpoint
        )
        self.maxpoollinear = MaxPoolLinear(i_ch=self.ch_h1, o_ch=self.ch_h1)
        self.lineardwconv = LinearDWConv(i_ch=self.ch_h2, o_ch=self.ch_h2)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        # divide x into x_l, x_h1, x_h2
        # x [B, HW, C]
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(-1, H, W, C).permute(0, 3, 1, 2) # [B, C, H, W]
        x_l = x[:, :self.ch_l, :, :].view(-1, self.ch_l, H*W)
        x_h1 = x[:, self.ch_l:(self.ch_l+self.ch_h1), :, :]
        x_h2 = x[:, (self.ch_l+self.ch_h1):, :, :]

        # operate x_l, x_h1, x_h2 separately
        x_l = self.swintransformer(x_l)
        x_h1 = self.maxpoollinear(x_h1)
        x_h2 = self.lineardwconv(x_h2)
        # print(x_l.shape, x_h1.shape, x_h2.shape)

        x = torch.cat((x_l, x_h1, x_h2), dim=1) # [B, C, H, W]
        x = x + self.dwconv(x) # [B, C, H, W]
        x = x.permute(0, 2, 3, 1) # [B, H, W, C]
        x = self.fc(x) # [B, H, W, C]
        x = x.view(B, H*W, C) # [B, HW, C]
        return x


class FeedForward(nn.Module):
    def __init__(self, d_io, d_hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_io, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_io)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class InceptionSwinTransformerBlock(nn.Module):
    '''
    Inception Swin Transformer Block, [B, H*W, C] -> [B, H*W, C]

    Args:
        dim (int): Number of input channels.
        alpha (float): Low frequency / high frequency.
        input_resolution (tuple[int]): Input resulotion.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.

    Returns:
        y = x + InceptionMixer(LN(x))
        z = y + FFN(LN(y))
    '''

    def __init__(
        self, dim, alpha, input_resolution, depth, num_heads, window_size,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False
    ):
        super().__init__()
        # operations
        self.norm1 = nn.LayerNorm(dim)
        self.inceptionmixer = InceptionMixer(
            dim, alpha, input_resolution, depth, num_heads, window_size,
            mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
            drop_path, norm_layer, downsample, use_checkpoint
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim*2)


    def forward(self, x):
        x = x + self.norm1(self.inceptionmixer(x))
        x = x + self.norm2(self.ffn(x))
        return x


class PatchEmbedding(nn.Module):
    '''
    Image to Patch Embedding: downsample patch_size times and norm
    [B, i_ch, H, W] proj(conv) -> [B, C, H/4, W/4] flatten+transpose -> [B, H/4*W/4, C]

    Args:
        img_size (int): Image size.  Default: 512.
        patch_size (int): Downsample times. Default: 4.
        i_ch (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None

    Returns:
        x = norm(conv(x)), 
    '''
    def __init__(self, img_size=512, patch_size=4, i_ch=3, embed_dim=96, norm_layer=None):
        super().__init__()
        # parameters
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.i_ch = i_ch
        self.embed_dim = embed_dim

        # operations
        self.proj = nn.Conv2d(i_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        '''
        x: [B, i_ch, H, W]
        '''
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, i_ch, H, W] -> [B, C, H/4, W/4] -> [B, C, H/4*W/4] -> [B, H/4*W/4, C]
        return self.norm(self.proj(x).flatten(2).transpose(1, 2))


class PatchMerging(nn.Module):
    '''
    Patch Merging Layer.Downsample
    [B, H*W, C] view+cat+view -> [B, H/2*W/2, 4*C] norm+reduction(Linear) -> [B, H/2*W/2, 2*C]

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    Returns:
        x = reduction(norm(cat(x0, x1, x2, x3)))
    '''
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        '''
        x: [B, H*W, C]
        '''
        # print('PatchMerging', x.shape)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        # print('PatchMerging end', x.shape)

        return x


class PatchExpanding(nn.Module):
    '''
    Patch Expanding Layer.Upsample 
    [B, H*W, C] expand(Linear) -> [B, H*W, 2*C] view+rearrange+view -> [B, H*2*W*2, C//2]

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        dim_scale: Expand dimention scale.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        
    Returns:
        x = norm(rearrange(expand(x)))
    '''

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        '''
        x: [B, H*W, C]
        '''
        # print('PatchExpanding', x.shape)
        H, W = self.input_resolution
        x = self.expand(x) # [B, H*W, 2*C]
        B, L, C = x.shape # [B, H*W, 2*C]
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B, -1, C // 4)  # [B, H*2*W*2, C//2]
        x = self.norm(x)
        # print('PatchExpanding end', x.shape)

        return x


class FinalPatchExpand_X4(nn.Module):
    '''
    final patch expanding layer
    [B, H*W, C] expand(Linear) -> [B, H*W, 16*C] view+rearrange+view -> [B, H*4*W*4, C]

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        dim_scale: Expand dimention scale.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    Returns:
        x = norm(rearrange(expand(x)))
    '''
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.output_dim = dim
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        '''
        x: [B, H*W, C]
        '''
        H, W = self.input_resolution
        x = self.expand(x) # [B, H*W, 16*C]
        B, L, C = x.shape # [B, H*W, 16*C]
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2)) # [B, H*4*W*4, C]
        x = x.view(B, -1, self.output_dim)  # [B, H*4*W*4, C]
        x = self.norm(x)

        return x


class Basiclayer_down(nn.Module):
    '''
    InceptionSwinTransformerBlock + downsample layer(PatchMerging)

    Args:
        dim (int): Number of input channels.
        alpha (float): Low frequency / high frequency.
        input_resolution (tuple[int]): Input resulotion.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.

    Returns:
        x = InceptionSwinTransformerBlock(x) * (depth // 2) # repeat depth // 2 times
        x = downsample(x)    
    '''
    def __init__(
        self, dim, alpha, input_resolution, depth, num_heads, window_size,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False
    ):
        super().__init__()
        # parameters
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # build InceptionSwinTransformerBlock blocks, [B, HW, C] -> [B, HW, C]
        self.blocks = nn.ModuleList([
            InceptionSwinTransformerBlock(
                dim, alpha, input_resolution, depth, num_heads, window_size,
                mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                drop_path, norm_layer, downsample, use_checkpoint
            ) for _ in range(depth // 2)
        ])

        # patch merging layer, [B, H*W, C] -> [B, H/2*W/2, 2*C]
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else nn.Identity()

    def forward(self, x):

        # print('Basiclayer_down', x.shape)
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x = self.downsample(x)
        # print('after Basiclayer_down', x.shape)

        return x


class Basiclayer_up(nn.Module):
    '''
    InceptionSwinTransformerBlock + upsample layer(PatchExpanding)

    Args:
        dim (int): Number of input channels.
        alpha (float): Low frequency / high frequency.
        input_resolution (tuple[int]): Input resulotion.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.

    Returns:
        x = InceptionSwinTransformerBlock(x) * (depth // 2) # repeat depth // 2 times
        x = upsample(x)    
    '''
    def __init__(
        self, dim, alpha, input_resolution, depth, num_heads, window_size,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False
    ):
        super().__init__()
        # parameters
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # build InceptionSwinTransformerBlock blocks, [B, H, W, C] -> [B, H, W, C]
        self.blocks = nn.ModuleList([
            InceptionSwinTransformerBlock(
                dim, alpha, input_resolution, depth, num_heads, window_size,
                mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                drop_path, norm_layer, upsample, use_checkpoint
            ) for _ in range(depth // 2)
        ])

        # patch expanding layer, [B, H*W, C] -> [B, H*2*W*2, C//2]
        self.upsample = upsample(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer) if upsample else nn.Identity()

    def forward(self, x):

        # print('Basiclayer_up', x.shape)
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        x = self.upsample(x)
        # print('after Basiclayer_up', x.shape)

        return x


class InceptionSwinUnet(nn.Module):
    '''
    Inception Swin Unet

    Args:
        alphas (List[float]): A list of low frequency/all channels. Default: [3/8, 4/8, 5/8, 6/8]
        img_size (int | tuple(int)): Input image size. Default 512
        patch_size (int | tuple(int)): Patch size. Default: 4
        i_ch (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96        
        depths (tuple(int)): Depth of each encoder layer InceptionSwinTransformerBlock. Default: [2, 2, 2, 2]
        depths_decoder (tuple(int)): Depth of each decoder layer InceptionSwinTransformerBlock. Default: [0, 2, 2, 2]
        num_heads (tuple(int)): Number of attention heads in different layers. Default: [3, 6, 12, 24]        
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None        
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1        
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False

    Returns:

        x = patch_embed(x)                          # [B, i_ch, H, W] -> [B, H/4*W/4, C]
        x = x + absolute_pos_embed                  # [B, H/4*W/4, C] -> [B, H/4*W/4, C]
        x = pos_drop(x)                             # [B, H/4*W/4, C] -> [B, H/4*W/4, C]

        # encoder ...
        x_dowmsamples.append(x)                     # [1] [[B, H/4*W/4, C]]
        x = InceptionSwinTransformerBlock(x)        # [B, H/4*W/4, C] -> [B, H/4*W/4, C]
        x = PatchMerging(x)                         # [B, H/4*W/4, C] -> [B, H/8*W/8, C*2]

        x_dowmsamples.append(x)                     # [2] [[B, H/4*W/4, C], [B, H/8*W/8, C*2]]
        x = InceptionSwinTransformerBlock(x)        # [B, H/8*W/8, C*2] -> [B, H/8*W/8, C*2]
        x = PatchMerging(x)                         # [B, H/8*W/8, C*2] -> [B, H/16*W/16, C*4]

        x_dowmsamples.append(x)                     # [3] [[B, H/4*W/4, C], [B, H/8*W/8, C*2], [B, H/16*W/16, C*4]]
        x = InceptionSwinTransformerBlock(x)        # [B, H/16*W/16, C*4] -> [B, H/16*W/16, C*4]
        x = PatchMerging(x)                         # [B, H/16*W/16, C*4] -> [B, H/32*W/32, C*8]

        x_dowmsamples.append(x)                     # [4] [[B, H/4*W/4, C], [B, H/8*W/8, C*2], [B, H/16*W/16, C*4], [B, H/32*W/32, C*8]]
        x = InceptionSwinTransformerBlock(x)        # [B, H/32*W/32, C*8] -> [B, H/32*W/32, C*8]

        x = norm_after_encoder()                    # [B, H/32*W/32, C*8] -> [B, H/32*W/32, C*8]

        # decoder ...
        x = PatchExpanding(x)                       # [B, H/32*W/32, C*8] -> [B, H/16*W/16, C*4]

        x = cat(x, x_dowmsamples[nlayers-1-i])      # [B, H/16*W/16, C*4]*2 -> [B, H/16*W/16, C*8]
        x = linear_after_cat(x)                     # [B, H/16*W/16, C*8] -> [B, H/16*W/16, C*4]
        x = InceptionSwinTransformerBlock(x)        # [B, H/16*W/16, C*4] -> [B, H/16*W/16, C*4]
        x = PatchExpanding(x)                       # [B, H/16*W/16, C*4] -> [B, H/8*W/8, C*2]

        x = cat(x, x_dowmsamples[nlayers-1-i])      # [B, H/8*W/8, C*2]*2 -> [B, H/8*W/8, C*4]
        x = linear_after_cat(x)                     # [B, H/8*W/8, C*4] -> [B, H/8*W/8, C*2]
        x = InceptionSwinTransformerBlock(x)        # [B, H/8*W/8, C*2] -> [B, H/8*W/8, C*2]
        x = PatchExpanding(x)                       # [B, H/8*W/8, C*2] -> [B, H/4*W/4, C]

        x = cat(x, x_dowmsamples[nlayers-1-i])      # [B, H/4*W/4, C]*2 -> [B, H/4*W/4, C*2]
        x = linear_after_cat(x)                     # [B, H/4*W/4, C*2] -> [B, H/4*W/4, C]
        x = InceptionSwinTransformerBlock(x)        # [B, H/4*W/4, C] -> [B, H/4*W/4, C]

        x = norm_after_decoder()                    # [B, H/4*W/4, C] -> [B, H/4*W/4, C]

        x = FinalPatchExpand_X4(x)+view+permute     # [B, H/4*W/4, C] -> [B, H*W, C] -> [B, H, W, C] -> [B, C, H, W]
        x = conv(x)                                 # [B, C, H, W] -> [B, num_classes, H, W]
    '''
    def __init__(
        self, alphas=[3/8, 4/8, 5/8, 6/8], img_size=512, patch_size=4, i_ch=3, num_classes=1000, embed_dim=96,
        depths=[2, 2, 2, 2], depths_decoder=[0, 2, 2, 2], num_heads=[3, 6, 12, 24],
        window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=.1, 
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, **kwargs
    ):
        super().__init__()
        depths_decoder = depths[::-1]
        depths_decoder[0] = 0
        print(f'InceptionSwinUnet initialization parameters----depths: {depths}, depths_decoder: {depths_decoder}, num_heads: {num_heads}, num_classes: {num_classes}')

        # parameters
        self.patch_norm = patch_norm
        self.ape = ape
        self.embed_dim = embed_dim
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 1.split image into non-overlapping patches
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, i_ch, embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        self.patches_resolution = patches_resolution = self.patch_embed.patches_resolution # H/4=512/4=128
        num_patches = self.patch_embed.num_patches # 128*128

        # 2.absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3.encoder
        self.layers_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_down = Basiclayer_down(
                dim=int(embed_dim * 2 ** i_layer), # C, C*2, C*4, C*8
                alpha=alphas[i_layer], # 3/8, 4/8, 5/8, 6/8
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[0] // (2 ** i_layer)
                ), # H/4, H/8, H/16, H/32
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint = use_checkpoint
            )
            self.layers_down.append(layer_down)

        # 4.decoder
        self.linears_after_cat = nn.ModuleList()
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):

            linear_after_cat = nn.Linear(
                2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            ) if i_layer > 0 else nn.Identity()
            self.linears_after_cat.append(linear_after_cat) # Identity, L(8C, 4C), L(4C, 2C), L(2C, C)

            if i_layer == 0:
                layer_up = PatchExpanding(
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))
                    ), # H/32
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), # 8C
                    dim_scale=2,
                    norm_layer=norm_layer
                )
            else:
                layer_up = Basiclayer_up(
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), # (8C), 4C, 2C, C
                    alpha=alphas[self.num_layers - 1 - i_layer], # None, 5/8, 4/8, 3/8
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer))
                    ), # (H/32), H/16, H/8, H/4
                    depth=depths[self.num_layers - 1 - i_layer],
                    num_heads=num_heads[self.num_layers - 1 - i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpanding if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint = use_checkpoint
                )
            self.layers_up.append(layer_up)
        
        # after 3 and after 4: norm layers
        self.norm_after_encoder = norm_layer(self.num_features)
        self.norm_after_decoder = norm_layer(self.embed_dim)

        # 5.final expand and linear projection
        self.final_expand = FinalPatchExpand_X4(
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            dim=embed_dim,
            dim_scale=4,
            norm_layer=norm_layer
        )
        self.final_conv = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        # self.activation = nn.Sigmoid()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Encoder backbone
    def forward_down(self, x):
        
        x_dowmsamples = []
        x = self.patch_embed(x)
        x = x + self.absolute_pos_embed if self.ape else x
        x = self.pos_drop(x)
        for layer_down in self.layers_down:
            x_dowmsamples.append(x)
            x = layer_down(x)
        x = self.norm_after_encoder(x)

        return x, x_dowmsamples

    # Decoder backbone
    def forward_up(self, x, x_dowmsamples):

        for idx, layer_up in enumerate(self.layers_up):
            if idx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_dowmsamples[self.num_layers - 1 - idx]], -1)
                x = self.linears_after_cat[idx](x)
                x = layer_up(x)
        x = self.norm_after_decoder(x)

        return x

    # Finally expanding + conv layer
    def forward_final(self, x):

        H, W = self.patches_resolution
        B, _, _ = x.shape
        x = self.final_expand(x) # [B, H/4*W/4, C] -> [B, H*W, C]
        x = x.view(B, 4*H, 4*W, -1) # [B, H*4, C] -> [B, H, W, C]
        x = x.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        x = self.final_conv(x) # [B, C, H, W] -> [B, class, H, W]

        return x

    def forward(self, x):
        
        x, x_dowmsamples = self.forward_down(x)
        x = self.forward_up(x, x_dowmsamples)
        x = self.forward_final(x)

        return x


class InceptionSwinUnet_from_config(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.inception_swin_unet = InceptionSwinUnet(
            alphas=config.ALPHAS,
            img_size=config.IMG_SIZE,
            patch_size=config.PATCH_SIZE,
            i_ch=config.IN_CHANS,
            num_classes=config.NUM_CLASSES,
            embed_dim=config.EMBED_DIM,
            depths=config.DEPTHS,
            num_heads=config.NUM_HEADS,
            window_size=config.WINDOW_SIZE,
            mlp_ratio=config.MLP_RATIO,
            qkv_bias=config.QKV_BIAS,
            qk_scale=config.QK_SCALE,
            drop_rate=config.DROP_RATE,
            drop_path_rate=config.DROP_PATH_RATE,
            ape=config.APE,
            patch_norm=config.PATCH_NORM,
            use_checkpoint=config.USE_CHECKPOINT
        )


    def forward(self, x):
        x = x.repeat(1,3,1,1) if x.size()[1] == 1 else x
        x = self.inception_swin_unet(x)
        return x


iswin_config1 = ml_collections.ConfigDict({
    'ALPHAS': [3/8, 4/8, 5/8, 6/8],
    'IMG_SIZE': 512,
    'PATCH_SIZE': 4,
    'IN_CHANS': 3,
    'NUM_CLASSES': 2,
    'EMBED_DIM': 96,
    'DEPTHS': [2, 2, 4, 2],
    'NUM_HEADS': [3, 6, 12, 24],
    'WINDOW_SIZE': 8,
    'MLP_RATIO': 4.,
    'QKV_BIAS': True,
    'QK_SCALE': None,
    'DROP_RATE': 0.0,
    'DROP_PATH_RATE': 0.1,
    'APE': False,
    'PATCH_NORM': True,
    'USE_CHECKPOINT': False
})

iswin_config = ml_collections.ConfigDict({
    'ALPHAS': [3/8, 4/8, 5/8],
    'IMG_SIZE': 512,
    'PATCH_SIZE': 4,
    'IN_CHANS': 3,
    'NUM_CLASSES': 2,
    'EMBED_DIM': 96,
    'DEPTHS': [2, 2, 4],
    'NUM_HEADS': [3, 6, 12],
    'WINDOW_SIZE': 8,
    'MLP_RATIO': 4.,
    'QKV_BIAS': True,
    'QK_SCALE': None,
    'DROP_RATE': 0.0,
    'DROP_PATH_RATE': 0.1,
    'APE': False,
    'PATCH_NORM': True,
    'USE_CHECKPOINT': False
})


if __name__ == '__main__':

    model = InceptionSwinUnet_from_config(config=iswin_config)

    # print(model)

    total_params = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total_params/1e6))
    
    res = model(torch.randn(12, 1, 512, 512))
    
    print(res.shape)

    res = model(torch.randn(8, 1, 512, 512))
    
    print(res.shape)