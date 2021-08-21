import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
from tqdm import trange
from collections import namedtuple

from features.car_body_feature import CarBodyFeature
from features.seat_material_feature import SeatMaterialFeature
from features.car_producer_feature import CarProducerFeature
from features.car_color_feature import CarColorFeature
from features.interior_color_feature import InteriorColorFeature

class CarAdDataset(Dataset):
    features = [#CarProducerFeature(),
                CarBodyFeature(),
                # SeatMaterialFeature(),
                # CarColorFeature(),
                # InteriorColorFeature()
                ]

    def __init__(self, csv_path):
        """
        Custom dataset example for reading image locations and labels from csv
        but reading images from files
        Args:
            csv_path (string): path to csv file
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)

        # data filtering
        mask = ~self.data_info['marka'].isin([None])
        for feature in self.features:
            mask &= feature.validDataMaskFromDF(self.data_info)

        self.data_info = self.data_info[mask]

        # self.data_info.loc[self.data_info['materijal_enterijera'].isin(['Prirodna koža', 'Kombinovana koža']), 'materijal_enterijera'] = 'Koža'
        # print(self.data_info['materijal_enterijera'].value_counts())
        max_size = self.data_info['karoserija'].value_counts().max()
        lst = [self.data_info]
        for class_index, group in self.data_info.groupby('karoserija'):
            lst.append(group.sample(max_size - len(group), replace=True))
        self.data_info = pd.concat(lst)

        for feature in self.features:
            feature.calculateGradWeight(self.data_info)

        self.data_info.to_csv('tmp_to_filter.csv', index=False)
        self.data_info = pd.read_csv('tmp_to_filter.csv', header=0)
        # First column contains the image paths
        self.ad_ids = np.asarray(self.data_info.iloc[:, 12].astype(str))
        # print(len(self.ad_ids))


        # print(self.ad_ids)
        # Second column is the labels
        # self.labels_str = np.asarray(self.data_info.iloc[:, 5])
        # self.labels = np.zeros(self.labels_str.shape)
        # self.labels = np.where(self.labels_str == 'Limuzina', 1, 0)

        # self.labels = [float(lbl[:-2].replace('.', '')) for lbl in self.labels]
        # Calculate len
        self.data_len = len(self.data_info.index)
        # self.data_info = self.data_info.reindex()
        # print(self.data_info[10:20])
        # exit(1)
        print(self.data_len)
        
    def __getitem__(self, index):
        # print(index)
        ad_id = self.ad_ids[index]
        # print(ad_id)

        fldr_pth = os.path.join('slike', ad_id)
        imgs = torch.FloatTensor(np.zeros((50, 3, 200, 200)))
        num_imgs = len(os.listdir(fldr_pth))
        i = 0
        img_sizes = np.zeros((50, 2))
        for file in os.listdir(fldr_pth):
            file_pth = os.path.join(fldr_pth, file)

            img_as_img = Image.open(file_pth)
            # print(img_as_img.size)
            if img_as_img.size[0] > img_as_img.size[1]:
                scale = 200/img_as_img.size[0]
            else:
                scale = 200/img_as_img.size[1]
            newsize = (int(round(img_as_img.size[0]*scale)), int(round(img_as_img.size[1]*scale)))
            img_sizes[i, 0] = newsize[0]
            img_sizes[i, 1] = newsize[1]
            # print(newsize)
            # newsize = (200, 150)
            img_as_img = img_as_img.resize(newsize)
            img_as_tensor = self.to_tensor(img_as_img)
            imgs[i, :, 0:newsize[1], 0:newsize[0]] = img_as_tensor
            # if img_as_img.size[0] > img_as_img.size[1]:
            #     pad1 = (200 - img_as_img.size[1])//2
            #     pad2 = 200 - img_as_img.size[1] - pad1
            #     imgs[i, :, pad1:200-pad2, :] = img_as_tensor
            # else:
            #     pad1 = (200 - img_as_img.size[0]) // 2
            #     pad2 = 200 - img_as_img.size[0] - pad1
            #     imgs[i, :, :, pad1:200 - pad2] = img_as_tensor

            # imgs[i, :] = img_as_tensor
            i += 1
            # print(i)
        # exit(1)

        lbls_dict = {}
        for feature in self.features:
            # print(index)
            # print(self.data_info[feature.name()][40:60])
            lbls_dict[feature.name()] = [feature.nameToClassId(self.data_info[feature.name()][index]),
                                         feature.getWeightForSample(self.data_info[feature.name()][index])]

            # print(lbls_dict[feature.name()])

        # print(len(lbls_dict))
        return [imgs, num_imgs, lbls_dict, img_sizes]

    def __len__(self):
        return self.data_len


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):

        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)

        return x, h


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


def main():
    ds = CarAdDataset('audi_a4.csv')

    loader_train = DataLoader(ds, batch_size=10,
                              shuffle=True, drop_last=True,
                              num_workers=1)

    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    resnet18_config = ResNetConfig(block=BasicBlock,
                                   n_blocks=[2, 2, 2, 2],
                                   channels=[64, 128, 256, 512])

    resnet18 = ResNet(resnet18_config)

    rnn = nn.LSTM(input_size=512,  # Added 2 * because keeping 2 times more channels
                  hidden_size=256,
                  num_layers=2,
                  dropout=0.5,
                  batch_first=False,
                  bidirectional=False)

    iterator_train = iter(loader_train)
    for i in trange(len(loader_train), desc='TeSt', position=1):
        next_data = next(iterator_train)

        image = next_data[0]

        # print(resnet18(image[0, :])[0].shape)
        features = resnet18(image[0, :])[0]
        print(features.shape)
        # print(next_data[1].numpy())
        # for i in range(next_data[1].numpy()[0]):
        len_first = next_data[1].numpy()[0]
        features = features[0:len_first, :]
        output, hidden = rnn(features.view(len_first, 1, 512))
        print(output.shape)


if __name__ == '__main__':
    main()
