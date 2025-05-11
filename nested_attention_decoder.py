import torch
import torch.nn as nn
import torch.nn.functional as F
from smt_encoder import SMT


class EfficientAttention(nn.Module):
    def __init__(self, channels, patch_size=14):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size

        # Linear transformations for Q, K, V
        self.query = nn.Conv2d(channels, channels, kernel_size=1)  # Q
        self.key = nn.Conv2d(channels, channels, kernel_size=1)    # K
        self.value = nn.Conv2d(channels, channels, kernel_size=1)  # V

    def forward(self, x, *skips):
        """
        Args:
            x: Input feature map from the decoder (B, C, H, W).
            *skips: Variable number of skip connection feature maps from the encoder.
        Returns:
            Output feature map after applying efficient attention to all skip connections.
        """
        B, C, H, W = x.shape

        print(f"Input shape: {x.shape}")
        for i, skip in enumerate(skips):
            print(f"Skip {i} shape: {skip.shape}")

        # Resize skip connections to match the spatial dimensions of x
        skips = [F.interpolate(skip, size=(H, W), mode='bilinear', align_corners=False) for skip in skips]

        # Split feature map into patches
        patch_H = H // self.patch_size
        patch_W = W // self.patch_size
        x_patches = x.view(B, C, patch_H, self.patch_size, patch_W, self.patch_size)
        x_patches = x_patches.permute(0, 2, 4, 1, 3, 5).reshape(B * patch_H * patch_W, C, self.patch_size, self.patch_size)

        # Initialize attention output
        attention_output = 0

        # Process each skip connection
        for skip in skips:
            # Split skip connection into patches
            skip_patches = skip.view(B, C, patch_H, self.patch_size, patch_W, self.patch_size)
            skip_patches = skip_patches.permute(0, 2, 4, 1, 3, 5).reshape(B * patch_H * patch_W, C, self.patch_size, self.patch_size)

            # Compute Q, K, V
            Q = self.query(x_patches)  # (B * patch_H * patch_W, C, patch_size, patch_size)
            K = self.key(skip_patches)  # (B * patch_H * patch_W, C, patch_size, patch_size)
            V = self.value(skip_patches)  # (B * patch_H * patch_W, C, patch_size, patch_size)

            # Reshape Q, K, V for matrix multiplication
            Q = Q.view(B * patch_H * patch_W, C, -1)  # (B * patch_H * patch_W, C, patch_size * patch_size)
            K = K.view(B * patch_H * patch_W, C, -1)  # (B * patch_H * patch_W, C, patch_size * patch_size)
            V = V.view(B * patch_H * patch_W, C, -1)  # (B * patch_H * patch_W, C, patch_size * patch_size)

            # Compute attention weights
            attn_weights = torch.bmm(Q.transpose(1, 2), K)  # (B * patch_H * patch_W, patch_size * patch_size, patch_size * patch_size)
            attn_weights = F.softmax(attn_weights, dim=-1)  # (B * patch_H * patch_W, patch_size * patch_size, patch_size * patch_size)

            # Apply attention to V
            attn_output = torch.bmm(attn_weights, V.transpose(1, 2))  # (B * patch_H * patch_W, patch_size * patch_size, C)
            attn_output = attn_output.transpose(1, 2)  # (B * patch_H * patch_W, C, patch_size * patch_size)

            # Accumulate attention output
            attention_output += attn_output

        # Reshape back to (B, C, H, W)
        attention_output = attention_output.view(B, patch_H, patch_W, C, self.patch_size, self.patch_size)
        attention_output = attention_output.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return attention_output

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class SMTUNetPlusPlus(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dims=[64, 128, 256, 512, 1024],  deep_supervision=True):
        super(SMTUNetPlusPlus, self).__init__()

        # SMT Encoder
        #self.encoder = SMT(in_chans=1, num_classes=num_classes, embed_dims=embed_dims)
        self.encoder = SMT(pretrain_img_size=224, in_chans=1, num_classes=4, 
              embed_dims=[64, 128, 256, 512, 1024], ca_num_heads=[2, 2, 2, 2, 2],
              sa_num_heads=[-1, -1, 4, 8, 16], depths=[2, 2, 4, 4, 2], ca_attentions=[1, 1, 1, 0, 0], num_stages=5)
        self.deep_supervision = deep_supervision
        

        # 1x1 Convolutions to Align Skip Connection Channels
        self.align4 = nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=1)  # Align x30 (512 -> 512)
        self.align3 = nn.Conv2d(embed_dims[2], embed_dims[2], kernel_size=1)  # Align x20 (256 -> 256)
        self.align2 = nn.Conv2d(embed_dims[1], embed_dims[1], kernel_size=1)  # Align x10 (128 -> 128)
        self.align1 = nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=1)  # Align x00 (64 -> 64)

# Encoder Blocks
        self.conv00 = ConvBlock(in_channels, 64)
        self.conv10 = ConvBlock(64, 128)
        self.conv20 = ConvBlock(128, 256)
        self.conv30 = ConvBlock(256, 512)
        self.conv40 = ConvBlock(512, 1024)

# Decoder Blocks (Nested Connections)
        self.conv01 = ConvBlock(64, 64)  # Adjusted input channels to 64
        self.conv11 = ConvBlock(128, 128)
        self.conv21 = ConvBlock(256, 256)
        self.conv31 = ConvBlock(512, 512)

        self.conv02 = ConvBlock(64, 64)
        self.conv12 = ConvBlock(128, 128)
        self.conv22 = ConvBlock(256, 256)

        self.conv03 = ConvBlock(64, 64)
        self.conv13 = ConvBlock(128, 128)

        self.conv04 = ConvBlock(64, 64)
        

# Upsampling Layers
        self.upconv10 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv20 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv30 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv40 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.upconv11 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv21 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv31 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.upconv12 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv22 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.upconv13 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

# Output Layers
        self.final1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(64, num_classes, kernel_size=1)


# Efficient Attention Modules
        self.attention01 = EfficientAttention(64)
        self.attention11 = EfficientAttention(128)
        self.attention21 = EfficientAttention(256)
        self.attention31 = EfficientAttention(512)

        self.attention02 = EfficientAttention(64)
        self.attention12 = EfficientAttention(128)
        self.attention22 = EfficientAttention(256)

        self.attention03 = EfficientAttention(64)
        self.attention13 = EfficientAttention(128)

        self.attention04 = EfficientAttention(64) 


    def forward(self, x):
        # Encoder Outputs
        encoder_outs = self.encoder(x)  # [x00, x10, x20, x30, x40]
        x00, x10, x20, x30, x40 = encoder_outs


        # Decoder Path
        x01 = self.conv01(self.attention01(self.align1(x00), self.upconv10(x10)))  # Level 0, Step 1
        x11 = self.conv11(self.attention11(self.align2(x10), self.upconv20(x20)))  # Level 1, Step 1
        x21 = self.conv21(self.attention21(self.align3(x20), self.upconv30(x30)))  # Level 2, Step 1
        x31 = self.conv31(self.attention31(self.align4(x30), self.upconv40(x40)))  # Level 3, Step 1


        x02 = self.conv02(self.attention02(self.align1(x00), x01, self.upconv11(x11)))  # Level 0, Step 2
        x12 = self.conv12(self.attention12(self.align2(x10), x11, self.upconv21(x21)))  # Level 1, Step 2
        x22 = self.conv22(self.attention22(self.align3(x20), x21, self.upconv31(x31)))  # Level 2, Step 2

        x03 = self.conv03(self.attention03(self.align1(x00), x01, x02, self.upconv12(x12)))  # Level 0, Step 3
        x13 = self.conv13(self.attention13(self.align2(x10), x11, x12, self.upconv22(x22)))  # Level 1, Step 3

        x04 = self.conv04(self.attention04(self.align1(x00), x01, x02, x03, self.upconv13(x13)))  # Level 0, Step 4


        # Deep Supervision
        if self.deep_supervision:
            output1 = self.final1(x01)
            output2 = self.final2(x02)
            output3 = self.final3(x03)
            output4 = self.final4(x04)
            return [output1, output2, output3, output4]
        else:
            output = self.final4(x04)
            return output

# Test the model
model = SMTUNetPlusPlus(in_channels=1, num_classes=9, deep_supervision=True)
x = torch.randn(1, 1, 224, 224)  # Batch size 1, single-channel image, 128x128 resolution
outputs = model(x)

# Print shapes of all outputs
for i, output in enumerate(outputs):
    print(f"Output {i+1} shape:", output.shape)
