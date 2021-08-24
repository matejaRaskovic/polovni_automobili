import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from einops import rearrange, repeat

ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50'
]

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
            nn.GELU(),
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
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
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

class Resnet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

        self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

    def forward(self, x):
        features = []

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        # x = self.avgpool(x)

        return x

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4


class CarAdModel(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone='resnet18'):
        super(CarAdModel, self).__init__()
        self.backbone = backbone  # for now we use only resnet18
        self.rnn_hidden_size = 512  # This could be larger for server-side network for improved accuracy

        self.output_vector_size = 40

        self.feature_extractor = Resnet()

        self.rnn_img_cols = nn.LSTM(input_size=3584,  # 512 for resnet 18
                                    hidden_size=self.rnn_hidden_size,
                                    num_layers=2,
                                    dropout=0.5,
                                    batch_first=True,
                                    bidirectional=True)

        self.rnn_imgs = nn.LSTM(input_size=2048,  # 512 for resnet 18
                           hidden_size=self.rnn_hidden_size,
                           num_layers=2,
                           dropout=0.5,
                           batch_first=True,
                           bidirectional=False)

        self.transformer = Transformer(dim=2048, depth=12, heads=8, dim_head=64, mlp_dim=1024, dropout=0.2)

        self.linear = nn.Linear(in_features=self.rnn_hidden_size, out_features=100)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x, img_sizes):
        x = self._prepare_x(x)
        img_sizes = img_sizes.cpu().detach().numpy().astype(int)
        # feature_grid = torch.zeros((x.shape[0], 512, 5, 5))
        rnn_input = torch.zeros((1, x.shape[0], 2048))
        for i in range(x.shape[0]):
            # print(self.feature_extractor(x[i:i+1, :, 0:img_sizes[i, 1], 0:img_sizes[i, 0]]).shape)
            # print(feature_grid[i:i+1, :].shape)
            # exit(1)
            # feature_grid[i:i+1, :] = self.feature_extractor(x[i:i+1, :, 0:img_sizes[i, 1], 0:img_sizes[i, 0]])
            sample_feature_grid = self.feature_extractor(x[i:i+1, :, 0:img_sizes[i, 1], 0:img_sizes[i, 0]])
            # print(sample_feature_grid.shape)
            sample_feature_grid = sample_feature_grid.view(
                (sample_feature_grid.shape[0], 7 * sample_feature_grid.shape[1], sample_feature_grid.shape[3], 1))
            sample_feature_grid = sample_feature_grid.view((sample_feature_grid.shape[0], sample_feature_grid.shape[2], sample_feature_grid.shape[1]))
            # sample_feature_grid = sample_feature_grid.to(x.device)
            # print(sample_feature_grid.shape)
            rnn_output, (ht, ct) = self.rnn_img_cols(sample_feature_grid)
            ht = ht.view((1, ht.shape[1], 2048))
            rnn_input[0, i, :] = ht
            # print(ht.shape)

        # feature_grid = feature_grid.view((feature_grid.shape[0], 5*feature_grid.shape[1], feature_grid.shape[2], 1))
        # feature_grid = feature_grid.view((feature_grid.shape[0], feature_grid.shape[2], feature_grid.shape[1]))
        # feature_grid = feature_grid.to(x.device)

        # rnn_output, (ht, ct) = self.rnn_img_cols(feature_grid)
        # ht = ht.view((1, ht.shape[1], 2048))
        # print(ht.shape)


        # img_features = self.feature_extractor(x)

        # print(img_features.shape)
        # feature_grid = torch.zeros((x.shape[0]))
        # print(img_sizes.shape)
        # for i in range()
        # exit(1)

        # rnn_input = feature_grid.view(feature_grid.shape[0], 1, 512)

        rnn_input = rnn_input.to(x.device)
        # # rnn_input = ht
        # rnn_output, (ht, ct) = self.rnn_imgs(rnn_input)
        # # print(ht.shape)
        # lin_input = torch.flatten(ht[-1])

        transformer_output = self.transformer(rnn_input)
        transformer_output = transformer_output.mean(dim=1)

        output = self.linear(transformer_output)

        output = output.view((10, 10))

        return output