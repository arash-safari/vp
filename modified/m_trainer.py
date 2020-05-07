# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:30:48 2020

@author: Arash
"""
from m_util import conf_parser, model_object_parser, get_model_type, get_path, load_part
from consts import PIXELSNAIL, VQVAE, TOP, BOTTOM, MIDDLE
from m_train_pixelsnail import train as train_pixelsnail
from m_train_vqvae import train as train_vqvae

from torch import optim, nn
import torch
from scheduler import CycleScheduler


def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def get_scheduler(lr, epoch, sched, optimizer, loader):
    scheduler = None
    if sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, lr, n_iter=len(loader) * epoch, momentum=None
        )
    return scheduler


def train(folder_name, loader, dataset_name, n_run, start_epoch=-1, end_epoch=-1, batch_size=-1, sched=None, device='cuda',
          size=256, lr=-1,
          amp=None):
    model_type = get_model_type(folder_name)
    _, train_params = conf_parser(dataset_name, n_run, folder_name)
    model = model_object_parser(dataset_name, n_run, folder_name)
    model = model.to(device)
    args = {}
    if start_epoch > 1:
        try:
            ckpt = load_part(model, start_epoch - 1, device)
        except RuntimeError as e:
            print('not find checkpoint {}'.format(start_epoch - 1))
            return 0

        args = ckpt['args']
        model = ckpt['model']
        if 'lr' in args:
            lr = args['lr']
        if 'batch' in args:
            batch_size = args['batch']
        if 'amp' in args:
            amp = args['amp']

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

    args['lr'] = lr
    args['batch'] = batch_size
    args['amp'] = amp

    optimizer = get_optimizer(model, lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    if model_type == PIXELSNAIL:

        for i in range(start_epoch, end_epoch):
            scheduler = get_scheduler(lr, end_epoch - start_epoch, sched, optimizer, loader)

            train_pixelsnail(i, loader, model, optimizer, scheduler, device)
            save_path = get_path(dataset_name, n_run, model, folder_name, checkpoint=i)
            torch.save(
                {'model': model.module.state_dict(), 'args': args},
                save_path,
            )

    elif model_type == VQVAE:
        scheduler = get_scheduler(args, sched, optimizer, loader)
        for i in range(start_epoch, end_epoch):
            train_vqvae(i, loader, model, optimizer, scheduler, device, dataset_name, n_run)
            save_path = get_path(dataset_name, n_run, model, folder_name, checkpoint=i)
            torch.save(model.state_dict(), save_path)
