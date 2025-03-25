import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from abc import ABC, abstractmethod

# Conv2d_BN: for convolutional layers and batch normalization
class Conv2d_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bn_weight_init=1, norm_cfg=None):
        super().__init__()

        # Convolutional layer definition
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False
        )

        # Batch Normalization layer definition
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Weight initialization
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, bn_weight_init)
        init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# Axial position embedding class
class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()
        # Position embedding layer, initialized to normal distribution
        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        # shape of x should be (B, C_qk, H) or (B, C_qk, W)
        B, C, N = x.shape
        # Perform position embedding interpolation to keep the size matching x
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x


# Sea_Attention module definition
class Sea_Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=2, activation=nn.ReLU, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = nn.Sequential(activation(), Conv2d_BN(self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = nn.Sequential(activation(), Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = nn.Sequential(activation(), Conv2d_BN(self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        # Fixed parameter ks -> kernel_size
        self.dwconv = Conv2d_BN(2 * self.dh, 2 * self.dh, kernel_size=3, stride=1, padding=1, dilation=1, groups=2 * self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.dh, dim, kernel_size=1, norm_cfg=norm_cfg)

    def forward(self, x):  
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        xx = xx.sigmoid() * qkv
        return xx


# use sinusoidal position embedding to encode time step (https://arxiv.org/abs/1706.03762)
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def time_emb(t, dim):
        """对时间进行正弦函数的编码，单一维度
       目标：让模型感知到输入x_t的时刻t
       实现方式：多种多样
       输入x：[B, C, H, W] x += temb 与空间无关的，也即每个空间位置（H, W）,都需要加上一个相同的时间编码向量[B, C]
       假设B=1 t=0.1
       1. 简单粗暴法
       temb = [0.1] * C -> [0.1, 0.1, 0.1, ……]
       x += temb.reshape(1, C, 1, 1)
       2. 类似绝对位置编码方式
       本代码实现方式
       3. 通过学习的方式（保证T是离散的0， 1， 2， 3，……，T）
       temb_learn = nn.Parameter(T+1, dim)
       x += temb_learn[t, :].reshape(1, C, 1, 1)
       
       
        Args:
            t (float): 时间，维度为[B]
            dim (int): 编码的维度

        Returns:
            torch.Tensor: 编码后的时间，维度为[B, dim]  输入是[B, C, H, W]
        """
        # 生成正弦编码
        # 把t映射到[0, 1000]
        t = t * 1000
        # 10000^k k=torch.linspace……
        freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)

        return torch.cat([sin_emb, cos_emb], dim=-1)
    
def label_emb(y, dim):
    """对类别标签进行编码，同样采用正弦编码

    Args:
        y (torch.Tensor): 图像标签，维度为[B] label:0-9
        dim (int): 编码的维度

    Returns:
    torch.Tensor: 编码后的标签，维度为[B, dim]
    """
    y = y * 1000

    freqs = torch.pow(10000, torch.linspace(0, 1, dim // 2)).to(y.device)
    sin_emb = torch.sin(y[:, None] / freqs)
    cos_emb = torch.cos(y[:, None] / freqs)

    return torch.cat([sin_emb, cos_emb], dim=-1)



# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# use GN for norm layer
def norm_layer(channels):
    return nn.GroupNorm(32, channels)


# Residual block
class ResidualBlock(TimestepBlock):
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # pojection for time step embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x

# 更换注意力嵌入


# upsample
class Upsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


# downsample
class Downsample(nn.Module):
    def __init__(self, channels, use_conv):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(stride=2, kernel_size=2)

    def forward(self, x):
        return self.op(x)


# high-dimensional feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExtractor, self).__init__()
        
        # 增加卷积层的深度
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels * 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_channels * 4, in_channels, kernel_size=3, padding=1)
        
        # 使用BatchNorm2d、LeakyReLU 和 Dropout
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels * 2)
        self.norm3 = nn.BatchNorm2d(out_channels * 4)
        self.norm4 = nn.BatchNorm2d(in_channels)
        
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout2d(p=0.25)  # Dropout2d 可以用来对特征图进行Dropout
        
    def forward(self, x):
        # 通过每一层卷积，并应用BatchNorm和LeakyReLU
        x1 = self.relu(self.norm1(self.conv1(x)))
        x2 = self.relu(self.norm2(self.conv2(x1)))
        x3 = self.relu(self.norm3(self.conv3(x2)))
        
        # Dropout
        x3 = self.dropout(x3)

        # Residual Connection
        x = self.relu(self.norm4(self.conv4(x3))) + x 
        
        return x # 改进


class ImprovedClassifier(nn.Module):
    def __init__(self, model_channels, image_w, image_h, num_classes):
        super(ImprovedClassifier, self).__init__()
        
        # 首先使用卷积层提取特征
        self.conv1 = nn.Conv2d(model_channels, 64, kernel_size=3, stride=2, padding=1)  # 输出：64, image_w/2, image_h/2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 输出：128, image_w/4, image_h/4
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 输出：256, image_w/8, image_h/8
        
        # BatchNorm 和 ReLU 激活
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * (image_w // 8) * (image_h // 8), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)  # 输出类别标签
        
        # 激活函数
        self.relu = nn.ReLU()

        # Softmax层
        self.softmax = nn.Softmax(dim=1)  # 计算每一行的 Softmax（对类进行归一化）
        
    def forward(self, x):
        # 卷积层提取特征
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        
        # 平展成一维
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  
        # 应用 Softmax，输出每个类别的概率分布
        class_label = self.softmax(x) # 最后输出类别
        return class_label


# The full UNet model with attention and timestep embedding
class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 2, 2),
            conv_resample=True,
            num_heads=4,
            num_classes = 10,
            image_w = 32,
            image_h = 32
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.image_w = image_w
        self.image_h = image_h

        # Initialize feature extractor (a small CNN)
        self.feature_extractor = FeatureExtractor(in_channels, model_channels) # 改进

        # Auxiliary Feature Classifier
        # self.classifier = ImprovedClassifier(model_channels, image_w, image_h, num_classes) # 改进
        
        # time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # self.label_embedding = nn.Embedding(self.num_classes, time_embed_dim) # 改进

        # down blocks
        self.down_blocks = nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1))
        ])
        down_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResidualBlock(ch, mult * model_channels, time_embed_dim, dropout)
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(Sea_Attention(ch, 64, num_heads)) # AttentionBlock(ch, num_heads=num_heads)
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level != len(channel_mult) - 1:  # don't use downsample for the last stage
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample)))
                down_block_chans.append(ch)
                ds *= 2

        # middle block
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            Sea_Attention(ch, 64, num_heads), # AttentionBlock(ch, num_heads=num_heads)
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )

        # up blocks
        self.up_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResidualBlock(
                        ch + down_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(Sea_Attention(ch, 64, num_heads)) # AttentionBlock(ch, num_heads=num_heads)
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample))
                    ds //= 2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            norm_layer(ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )


    def forward(self, x, timesteps, labels=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x H x W] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param labels: a 1-D batch of labels.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        if isinstance(timesteps, (int, float)):
           timesteps = torch.tensor([timesteps], dtype=torch.float32, device=x.device)

        # Extract features from input using the FeatureExtractor
        x = self.feature_extractor(x) # 改进

        # time step embedding
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels)) # torch.Size([batch_size, 512])
        # emb = time_emb(timesteps, self.model_channels*4) # 正弦时间编码
        
        if labels is not None:
            labels_emb = label_emb(labels, self.model_channels*4)
            emb = emb + labels_emb # 将标签生成标签特征向量后与时间流特征向量直接加和，式模型获得时空和类别感知, torch.Size([batch_size, 512])
        
        h = x # torch.Size([batch_size, 3, 32, 32])
        
        # down stage
        for module in self.down_blocks:
            h = module(h, emb)
            hs.append(h)
        # middle stage
        h = self.middle_block(h, emb)
        # up stage
        for module in self.up_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb) # 实现unet编码器和解码器之间各层的连接
        return self.out(h) # self.classifier(h)
        