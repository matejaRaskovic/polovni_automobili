import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50'
]


class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 1), padding=ks//2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),  # Use this to go back to height scale 8
            ConvCompressH(in_c//2, out_c),
        )

    def forward(self, x):
        return self.layer(x)

class Resnet(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

        self.avgpool = nn.AdaptiveAvgPool2d((50, 50))

    def forward(self, x):
        features = []

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        feat_maps = []
        x = self.encoder.layer1(x)
        feat_maps.append(x)
        x = self.encoder.layer2(x)
        feat_maps.append(self.avgpool(x))
        x = self.encoder.layer3(x)
        feat_maps.append(self.avgpool(x))
        x = self.encoder.layer4(x)
        feat_maps.append(self.avgpool(x))

        return torch.cat(feat_maps, dim=1)

        # x = self.avgpool(x)
        #
        # return x

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

        self.feature_extractor = Resnet(backbone)

        self.rnn_img_cols = nn.LSTM(input_size=2560,  # 512 for resnet 18
                                    hidden_size=self.rnn_hidden_size,
                                    num_layers=2,
                                    dropout=0.5,
                                    batch_first=True,
                                    bidirectional=True)

        self.rnn_imgs = nn.LSTM(input_size=2048,  # 512 for resnet 18
                           hidden_size=self.rnn_hidden_size//2,
                           num_layers=2,
                           dropout=0.5,
                           batch_first=True,
                           bidirectional=True)

        self.linear = nn.Linear(in_features=2*self.rnn_hidden_size//2, out_features=100)

        self.ghc = GlobalHeightConv(960, 960//4)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x, img_sizes):
        x = self._prepare_x(x)
        img_sizes = img_sizes.cpu().detach().numpy().astype(int)
        # feature_grid = torch.zeros((x.shape[0], 512, 5, 5))
        # for i in range(x.shape[0]):
            # print(self.feature_extractor(x[i:i+1, :, 0:img_sizes[i, 1], 0:img_sizes[i, 0]]).shape)
            # print(feature_grid[i:i+1, :].shape)
            # exit(1)
            # feature_grid[i:i+1, :] = self.feature_extractor(x[i:i+1, :, 0:img_sizes[i, 1], 0:img_sizes[i, 0]])

        feature_grid = self.feature_extractor(x)

        feature_grid = feature_grid.view((feature_grid.shape[0], 5*feature_grid.shape[1], feature_grid.shape[2], 1))
        feature_grid = feature_grid.view((feature_grid.shape[0], feature_grid.shape[2], feature_grid.shape[1]))
        feature_grid = feature_grid.to(x.device)

        rnn_output, (ht, ct) = self.rnn_img_cols(feature_grid)
        ht = ht.view((1, ht.shape[1], 2048))
        # print(ht.shape)


        # img_features = self.feature_extractor(x)

        # print(img_features.shape)
        # feature_grid = torch.zeros((x.shape[0]))
        # print(img_sizes.shape)
        # for i in range()
        # exit(1)

        # rnn_input = feature_grid.view(feature_grid.shape[0], 1, 512)

        rnn_input = ht
        rnn_in_multiple = torch.zeros((3, rnn_input.shape[1], rnn_input.shape[2])).to(rnn_input.device)
        for i in range(3):
            rnn_in_multiple[i:i+1] = rnn_input[:, torch.randperm(rnn_input.shape[1])]

        rnn_output, (ht, ct) = self.rnn_imgs(rnn_in_multiple)
        ht_tmp = ht[-2:].transpose(2, 1)
        avg_pool = nn.AdaptiveAvgPool1d(1)
        lin_input = torch.flatten(avg_pool(ht_tmp))
        # print(ht.shape)
        # lin_input = torch.flatten(ht[-2:])
        output = self.linear(lin_input)

        output = output.view((10, 10))

        return output
