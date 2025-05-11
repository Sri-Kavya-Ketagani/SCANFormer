import math
import torch
import torch.nn as nn
from functools import partial
import torch.utils.checkpoint as checkpoint

# Helper function
def to_2tuple(x):
    """Convert a scalar to a tuple of (x, x) if it's not already a tuple."""
    if isinstance(x, tuple):
        return x
    return (x, x)

# DWConv Class
class DWConv(nn.Module):
    """Depthwise Convolution."""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)  # Depthwise convolution

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)  # Reshape to (B, C, H, W)
        x = self.dwconv(x)  # Apply depthwise convolution
        x = x.flatten(2).transpose(1, 2)  # Reshape back to (B, N, C)
        return x

# Mlp Class
class Mlp(nn.Module):
    """Multilayer perceptron with two fully connected layers and DWConv."""
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)  # Add DWConv here
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)  # Apply DWConv before activation
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Attention Class
class Attention(nn.Module):
    """Custom attention mechanism."""
    def __init__(self, dim, ca_num_heads, sa_num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()
        self.ca_attention = ca_attention
        self.sa_attention = 1 - ca_attention
        self.num_heads = ca_num_heads if ca_attention else sa_num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = qk_scale or (dim // self.num_heads) ** -0.5

    def forward(self, x, H, W):
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

# Block Class
class Block(nn.Module):
    """Basic block of the SMT with attention and MLP."""
    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1, expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, ca_num_heads, sa_num_heads, qkv_bias, qk_scale, attn_drop, drop, ca_attention, expand_ratio)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        def forward_fn(input_tensor):
             return self.drop_path(self.mlp(self.norm2(input_tensor), H, W))

        x = x + checkpoint.checkpoint(forward_fn, self.norm1(x), use_reentrant=False)
        
        return x

# OverlapPatchEmbed Class
class OverlapPatchEmbed(nn.Module):
    """Patch embedding with overlapping patches."""
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)  # Flatten height and width
        x = self.norm(x)
        return x, H // self.proj.stride[0], W // self.proj.stride[0]

# Head Class
class Head(nn.Module):
    """Initial convolutional head for the first stage."""
    def __init__(self, embed_dim, head_conv=3):
        super().__init__()
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=head_conv, stride=1, padding=head_conv // 2)  # Change to 1 input channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x).flatten(2).transpose(1, 2)
        return x, H, W


# SMT Class
class SMT(nn.Module):
    """Spatial Multi-head Transformer (SMT)."""
    def __init__(self, pretrain_img_size=224, in_chans=1, num_classes=6, embed_dims=[64, 128, 256, 512, 1024], ca_num_heads=[4, 4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 8, 16], mlp_ratios=[4, 4, 4, 4, 2], qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 6, 3], ca_attentions=[1, 1, 1, 1, 0], num_stages=5, head_conv=3, expand_ratio=2):
        super(SMT, self).__init__()
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Head(embed_dims[0], head_conv)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=in_chans if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i],
                ca_num_heads=ca_num_heads[i],
                sa_num_heads=sa_num_heads[i],
                mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + j],
                norm_layer=norm_layer,
                ca_attention=(1 if j % 2 == 0 else 0) if i in [2, 3] else ca_attentions[i],
                expand_ratio=expand_ratio)
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return tuple(outs)


import torch

# Define random input
batch_size = 1  # Number of samples
in_channels = 1  # Input image channels (e.g., RGB = 3)
image_size = 224  # Input image size (224x224 for typical models)

# Create random input tensor
x = torch.randn(batch_size, in_channels, image_size, image_size)

model = SMT(
    pretrain_img_size=224,
    in_chans=1,
    num_classes=6,
    embed_dims=[64, 128, 256, 512],  # Fewer dimensions
    ca_num_heads=[2, 2, 2, 2],       # Fewer heads
    sa_num_heads=[-1, -1, 4, 8],     # Reduce spatial attention heads
    depths=[2, 2, 4, 4],             # Shallower model
    ca_attentions=[1, 1, 1, 0],
    num_stages=4                     # Use fewer stages
)


# Forward pass
outputs = model(x)

# Print the outputs from each stage
for i, output in enumerate(outputs):
    print(f"Output from stage {i + 1}: shape {output.shape}")
