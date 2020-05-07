# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:30:48 2020

@author: Arash
"""
from m_util import conf_parser, model_object_parser, get_model_type, get_path, load_part
from consts import PIXELSNAIL, VQVAE, TOP, BOTTOM, MIDDLE
from m_train_pixelsnail import train as train_pixelsnail
from m_train_vqvae import train as train_vqvae
from torch.utils.data import DataLoader
from dataset import LMDBDataset
from torch import optim, nn
import torch
from scheduler import CycleScheduler
from torchvision import datasets, transforms


def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def get_scheduler(args, sched, optimizer, loader):
    scheduler = None
    if args.sched == 'cycle' or sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )
    return scheduler


def train(dataset_path, n_run, folder_name, start_epoch=-1, end_epoch=-1, batch_size=-1, sched=None, device='cuda',
          size=256, lr=-1,
          amp=None):
    model_type = get_model_type(folder_name)
    _, train_params = conf_parser(dataset_path, n_run, folder_name)
    model = model_object_parser(dataset_path, n_run, folder_name)
    model = model.to(device)
    if lr < 0:
        lr = train_params['lr']
    if start_epoch < 0:
        start_epoch = 0
    if end_epoch < 0:
        end_epoch = train_params['epoch']
    if batch_size < 0:
        batch_size = train_params['batch']
    if amp is None and 'amp' in train_params:
        amp = train_params['amp']

    optimizer = get_optimizer(model, lr)
    args = {'hier': folder_name}

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp)

    if start_epoch is not None:
        ckpt = load_part(model, start_epoch - 1, device)
        args = ckpt['args']
        model = ckpt['model']
    model = nn.DataParallel(model)
    model = model.to(device)

    if model_type == PIXELSNAIL:
        lampda_name = '{}_{}'.format(*[dataset_path, n_run])
        dataset = LMDBDataset(lampda_name)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
        )
        scheduler = get_scheduler(args, sched, optimizer, loader)

        for i in range(start_epoch, end_epoch):
            train_pixelsnail(args, i, loader, model, optimizer, scheduler, device)
            save_path = get_path(dataset, n_run, model, folder_name, checkpoint=i)
            torch.save(
                {'model': model.module.state_dict(), 'args': args},
                save_path,
            )

    elif model_type == VQVAE:
        transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        scheduler = get_scheduler(args, sched, optimizer, loader)
        for i in range(start_epoch, end_epoch):
            train_vqvae(i, loader, model, optimizer, scheduler, device, dataset_path, n_run)
            save_path = get_path(dataset, n_run, model, folder_name, checkpoint=i)
            torch.save(model.state_dict(), save_path)
