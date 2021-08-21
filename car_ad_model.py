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

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

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

        self.rnn = nn.LSTM(input_size=9*512,  # 512 for resnet 18
                           hidden_size=self.rnn_hidden_size,
                           num_layers=2,
                           dropout=0.5,
                           batch_first=False,
                           bidirectional=False)

        self.linear = nn.Linear(in_features=self.rnn_hidden_size, out_features=100)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std


    def forward(self, x):
        x = self._prepare_x(x)
        img_features = self.feature_extractor(x)

        rnn_input = img_features.view(img_features.shape[0], 1, 9*512)

        rnn_output, (ht, ct) = self.rnn(rnn_input)
        lin_input = torch.flatten(ht[-1])
        output = self.linear(lin_input)

        output = output.view((10, 10))

        return output