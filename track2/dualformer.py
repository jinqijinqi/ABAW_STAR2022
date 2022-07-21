"""
Code from
https://github.com/zengqunzhao/Former-DFER
"""
from einops import rearrange, repeat
from torch import nn, einsum
import math
import torch
from torchvision import models
from torch.functional import F
import numpy as np
from collections import OrderedDict
import hrnet

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_backbone(nn.Module):   # resnet18
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNet_backbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or"
                             " a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.layer5 = self._make_layer(block, 512, layers[3], stride=1, dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        #torch.Size([2, 16, 3, 112, 112])
        b,c,h,w = x.shape
        x = x.contiguous().view(-1, c, h, w)
        #torch.Size([32, 3, 112, 112])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # torch.Size([1, 64, 28, 28])
        x = self.layer1(x)     # torch.Size([1, 64, 28, 28])
        x = self.layer2(x)     # torch.Size([1, 128, 14, 14])
        x = self.layer3(x)     # torch.Size([1, 256, 7, 7])
        x = self.layer4(x)

        return x

def get_pretrain_resnet(pretrain_path):
    model = ResNet_backbone(BasicBlock, [2, 2, 2, 2])
    print ('loading pretrained model {}'.format(pretrain_path))
    pretrain = torch.load(pretrain_path)
    #print(pretrain.keys())
    net_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain.items() if k in net_dict.keys()}
    #print(pretrain_dict.keys())
    model.load_state_dict(pretrain_dict,strict=False)


    for name, value in model.named_parameters():
        if name.startswith('layer.'):
            value.requires_grad = False
    
    return model

class DualFormer(nn.Module):   # S-Former after stage3
    def __init__(self, resnet_weight, hrnet_weight, num_patches=7*7, dim=512, depth=2, heads=8, mlp_dim=512, dim_head=32, dropout=0.2,class_num=6):
        super(DualFormer, self).__init__()
        self.resnet = get_pretrain_resnet(resnet_weight)
        self.hrnet = hrnet.hrnet18(hrnet_weight,classify=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)).cuda()
        self.pos_embedding = torch.cat((nn.Parameter(torch.randn(1, num_patches + 1, dim)), nn.Parameter(torch.randn(1, num_patches + 1, dim))), dim=1).cuda()
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.classifier = nn.Linear(512, class_num)

    def forward(self, x):
        res_feature = self.resnet(x)
        hr_feature = self.hrnet(x)
        b_l, c, h, w = res_feature.shape
        #torch.Size([32, 256, 7, 7])
        res_feature = res_feature.reshape((b_l, c, h*w))
        res_feature = res_feature.permute(0, 2, 1)
        b, n, _ = res_feature.shape

        hr_feature = hr_feature.reshape((b_l, c, h*w))
        hr_feature = hr_feature.permute(0, 2, 1)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        #x shape: torch.Size([32, 49, 256])
        #pos_embedding shape torch.Size([1, 49, 256])
        x = torch.cat((cls_tokens, res_feature, cls_tokens, hr_feature), dim=1)
        x = x + self.pos_embedding
        #torch.Size([32, 49, 256])
        x = self.spatial_transformer(x)
        #torch.Size([32, 49, 256])
        x = x[:, 0]

        return self.classifier(x)