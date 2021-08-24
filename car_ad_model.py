import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50'
]


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

        x = self.avgpool(x)

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

        self.rnn_img_cols = nn.LSTM(input_size=2560,  # 512 for resnet 18
                                    hidden_size=2*self.rnn_hidden_size,
                                    num_layers=2,
                                    dropout=0.5,
                                    batch_first=True,
                                    bidirectional=True)

        self.rnn_imgs = nn.LSTM(input_size=2*2048,  # 512 for resnet 18
                           hidden_size=self.rnn_hidden_size,
                           num_layers=2,
                           dropout=0.5,
                           batch_first=True,
                           bidirectional=False)

        self.linear = nn.Linear(in_features=self.rnn_hidden_size, out_features=100)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x, img_sizes):
        x = self._prepare_x(x)
        img_sizes = img_sizes.cpu().detach().numpy().astype(int)
        feature_grid = torch.zeros((x.shape[0], 512, 5, 5))
        for i in range(x.shape[0]):
            # print(self.feature_extractor(x[i:i+1, :, 0:img_sizes[i, 1], 0:img_sizes[i, 0]]).shape)
            # print(feature_grid[i:i+1, :].shape)
            # exit(1)
            feature_grid[i:i+1, :] = self.feature_extractor(x[i:i+1, :, 0:img_sizes[i, 1], 0:img_sizes[i, 0]])

        feature_grid = feature_grid.view((feature_grid.shape[0], 10*feature_grid.shape[1], feature_grid.shape[2], 1))
        feature_grid = feature_grid.view((feature_grid.shape[0], feature_grid.shape[2], feature_grid.shape[1]))
        feature_grid = feature_grid.to(x.device)

        rnn_output, (ht, ct) = self.rnn_img_cols(feature_grid)
        ht = ht.view((1, ht.shape[1], 2*2048))
        # print(ht.shape)


        # img_features = self.feature_extractor(x)

        # print(img_features.shape)
        # feature_grid = torch.zeros((x.shape[0]))
        # print(img_sizes.shape)
        # for i in range()
        # exit(1)

        # rnn_input = feature_grid.view(feature_grid.shape[0], 1, 512)

        rnn_input = ht
        rnn_output, (ht, ct) = self.rnn_imgs(rnn_input)
        # print(ht.shape)
        lin_input = torch.flatten(ht[-1])
        output = self.linear(lin_input)

        output = output.view((10, 10))

        return output