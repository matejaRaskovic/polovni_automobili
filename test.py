# added this so that we can import from the parent dir
import os

import argparse
import numpy as np
from tqdm import trange

from torch.utils.data import DataLoader
import torch

from car_ad_model import CarAdModel
from automobili_dataset import CarAdDataset

torch.backends.cudnn.deterministic = True


def feed_forward(net, images, num_imgs, img_sizes, labels, device):
    images = images.to(device)
    img_sizes = img_sizes.to(device)

    total_loss = 0

    conf_mats = {}

    for i in range(images.shape[0]):
        imgs_for_sample = images[i, :]  # using only one sample from batch
        imgs_for_sample = imgs_for_sample[0:num_imgs[i], :]  # keeping only valid images - removing the padding
        img_sizes_for_sample = img_sizes[i, :]
        img_sizes_for_sample = img_sizes_for_sample[0:num_imgs[i], :]
        est = net(imgs_for_sample, img_sizes_for_sample)
        loss = 0
        for feature in CarAdDataset.features:
            vec = est[feature.pos():feature.pos()+1]
            loss += feature.calculateLoss(vec, labels[feature.name()][0][i], labels[feature.name()][1][i], device)
            conf_mat = feature.getConfMat(vec, labels[feature.name()][0][i])
            if feature.name() in conf_mats:
                conf_mats[feature.name()] += conf_mat
            else:
                conf_mats[feature.name()] = conf_mat

        total_loss += loss

    return total_loss, conf_mats


def conf_mat_to_samples(conf_mat):
    gt = []
    est = []
    print(conf_mat)
    for i in range(conf_mat.shape[0]):
        print(i)
        for j in range(conf_mat.shape[1]):
            print(j)
            for k in range(conf_mat[i][j]):
                gt.append(i)
                est.append(j)

    return gt, est


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', default=None,
                        help='path to load saved checkpoint.'
                             '(finetuning)')
    # Dataset related arguments
    parser.add_argument('--test_csv', default=None,
                        help='path to the csv containing train ads')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='numbers of workers for dataloaders')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='select gpu')
    parser.add_argument('--output_txt', default=None,
                        help='path to the output txt file')

    args = parser.parse_args()

    arg_device = 'cpu' if args.no_cuda else 'cuda:' + str(args.gpu_id)
    device = torch.device('cpu' if args.no_cuda else 'cuda:' + str(args.gpu_id))

    # Create dataloader
    dataset_test = CarAdDataset(args.test_csv)  # FLIPPING CAN BE ADDED
    loader_test = DataLoader(dataset_test, 5,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers,
                              pin_memory=not args.no_cuda,
                              worker_init_fn=lambda x: np.random.seed())

    net = CarAdModel('resnet34').to(device)
    net.load_state_dict(torch.load(args.pth, map_location=device))

    conf_mats = {}
    net.train()

    iterator_test = iter(loader_test)
    total_test_loss = 0

    for i in trange(len(loader_test),
                    desc='Test', position=2):

        try:
            next_data = next(iterator_test)
        except AssertionError:
            continue
        images = next_data[0]
        num_imgs = next_data[1]
        labels = next_data[2]
        img_sizes = next_data[3]

        with torch.cuda.amp.autocast() and torch.no_grad():
            loss, c_mats = feed_forward(net, images, num_imgs, img_sizes, labels, device)

        for key in c_mats:
            if key in conf_mats:
                conf_mats[key] += c_mats[key]
            else:
                conf_mats[key] = c_mats[key]
            # for debugging
            print(key)
            print(conf_mats[key])

        total_test_loss += loss.cpu().detach()

    # HERE WE HAVE TO CALCULATE METRICS
    keys = [k for k in c_mats]
    print(keys)
    gt, est = conf_mat_to_samples(conf_mats[keys[0]])
    print(gt)


if __name__ == '__main__':
    main()
