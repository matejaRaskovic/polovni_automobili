# added this so that we can import from the parent dir
import os, sys

import argparse
import numpy as np
from tqdm import trange

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch

from car_ad_model import CarAdModel
from automobili_dataset import CarAdDataset

torch.backends.cudnn.deterministic = True


def feed_forward(net, images, num_imgs, labels, device):
    images = images.to(device)
    lossFun = nn.L1Loss()
    labels = torch.from_numpy(np.array(labels)).to(device)

    total_loss = 0
    # print(images.shape[0])
    # print(labels)
    for i in range(images.shape[0]):
        imgs_for_sample = images[i, :]  # using only one sample from batch
        imgs_for_sample = imgs_for_sample[0:num_imgs[i], :]  # keeping only valid images - removing the padding

        est = net(imgs_for_sample)
        loss = lossFun(est, labels[i])
        total_loss += loss

    return total_loss


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--id', required=True,
                        help='experiment id to name checkpoints and logs')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--pth', default=None,
                        help='path to load saved checkpoint.'
                             '(finetuning)')
    # Model related
    parser.add_argument('--feature_extractor', default='resnet18',
                        choices=['resnet18', 'resnet50', 'resnet101', 'densenet121', 'resnext50_32x4d'],
                        help='backbone of the network')
    # Dataset related arguments
    parser.add_argument('--train_csv', default=None,
                        help='path to the csv containing train ads')
    parser.add_argument('--valid_csv', default='data/valid/',
                        help='path to the csv containing valid ads')
    parser.add_argument('--no_flip', action='store_true',
                        help='disable left-right flip augmentation')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='numbers of workers for dataloaders')
    # optimization related arguments
    parser.add_argument('--batch_size_train', default=8, type=int,
                        help='training mini-batch size')
    parser.add_argument('--batch_size_valid', default=2, type=int,
                        help='validation mini-batch size')
    parser.add_argument('--epochs', default=300, type=int,
                        help='epochs to train')
    parser.add_argument('--optim', default='Adam',
                        help='optimizer to use. only support SGD and Adam')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='factor for L2 regularization')
    # Misc arguments
    parser.add_argument('--no_cuda', action='store_true',
                        help='disable cuda')
    parser.add_argument('--seed', default=594277, type=int,
                        help='manual seed')
    parser.add_argument('--disp_iter', type=int, default=1,
                        help='iterations frequency to display')
    parser.add_argument('--save_every', type=int, default=25,
                        help='epochs frequency to save state_dict')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='select gpu')

    args = parser.parse_args()

    arg_device = 'cpu' if args.no_cuda else 'cuda:' + str(args.gpu_id)
    device = torch.device('cpu' if args.no_cuda else 'cuda:' + str(args.gpu_id))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(args.ckpt, args.id), exist_ok=True)

    # Create dataloader
    dataset_train = CarAdDataset(args.train_csv)  # FLIPPING CAN BE ADDED
    loader_train = DataLoader(dataset_train, args.batch_size_train,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers,
                              pin_memory=not args.no_cuda)
    if args.valid_csv:
        dataset_valid = CarAdDataset(args.valid_csv)
        loader_valid = DataLoader(dataset_valid, args.batch_size_valid,
                                  shuffle=False, drop_last=False,
                                  num_workers=args.num_workers,
                                  pin_memory=not args.no_cuda)

    # Create model
    if args.pth is not None:
        tmp = 7
        # We should implement loading of previously trained model for fine-tuning
    else:
        net = CarAdModel(args.feature_extractor).to(device)

    parameters_to_be_trained = filter(lambda p: p.requires_grad, net.parameters())

    # Create optimizer
    if args.optim == 'SGD':
        optimizer = optim.SGD(
            parameters_to_be_trained,
            lr=args.lr, momentum=args.beta1, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(
            parameters_to_be_trained,
            lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
    else:
        raise NotImplementedError()

    # Create grad scaler used for automatic mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Init variable
    args.max_iters = args.epochs * len(loader_train)
    args.cur_iter = 0
    args.best_valid_loss = 1e9

    # Start training
    for ith_epoch in trange(1, args.epochs + 1, desc='Epoch', unit='ep'):

        # Train phase
        device = torch.device(arg_device)
        net.train()
        iterator_train = iter(loader_train)
        for i in trange(len(loader_train),
                        desc='Train ep%s' % ith_epoch, position=1):

            args.cur_iter += 1
            try:
                next_data = next(iterator_train)
            except AssertionError:
                continue
            images = next_data[0]
            num_imgs = next_data[1]
            labels = next_data[2]

            with torch.cuda.amp.autocast():
                loss = feed_forward(net, images, num_imgs, labels, device)
                print(loss)

            scaler.scale(loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            nn.utils.clip_grad_norm_(net.parameters(), 3.0, norm_type='inf')

            # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
            # although it still skips optimizer.step() if the gradients contain infs or NaNs.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()

        # Valid phase
        net.eval()
        if args.valid_csv:
            iterator_valid = iter(loader_valid)
            total_valid_loss = 0
            valid_num = 0
            for i in trange(len(loader_valid),
                            desc='Valid ep%d' % ith_epoch, position=2):

                try:
                    next_data = next(iterator_valid)
                except AssertionError:
                    continue
                images = next_data[0]
                num_imgs = next_data[1]
                labels = next_data[2]

                with torch.cuda.amp.autocast():
                    loss = feed_forward(net, images, num_imgs, labels, device)
                    total_valid_loss += loss
                    print(total_valid_loss)

            # Save best validation loss model
            if total_valid_loss < args.best_valid_loss:
                args.best_valid_loss = total_valid_loss
                torch.save(net.state_dict(), os.path.join(args.ckpt, args.id, 'best_valid.pth'))  # this is temporary

        if ith_epoch % args.save_every == 0:
            torch.save(net.state_dict(),
                       os.path.join(args.ckpt, args.id, 'epoch_%d.pth' % ith_epoch))  # this is temporary


if __name__ == '__main__':
    main()
