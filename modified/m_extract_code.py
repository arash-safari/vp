import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm
from m_util import get_path
from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img, _, filename in pbar:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


def main(dataset_path, n_run, vqvae_ckpt, size, device='cuda'):
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = ImageFileDataset(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    ckpt_path = get_path(dataset, n_run, 'vqvae', 'ckpt', checkpoint=vqvae_ckpt)
    model = VQVAE()
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024
    lamda_name = '{}_{}'.format(*[dataset_path, n_run])
    env = lmdb.open(lamda_name, map_size=map_size)

    extract(env, loader, model, device)
