import sys

sys.path.append('../image/modified')
import torch
import lmdb
from image.modified.m_extract_vqvae import extract_videomnist


def extract(model, ckpt_path, lamda_name, device, loader):
    map_size = 100 * 1024 * 1024 * 1024
    model.load_state_dict(torch.load(ckpt_path), strict=False)
    model = model.to(device)
    model.eval()
    env = lmdb.open(lamda_name, map_size=map_size)
    extract_videomnist(env, loader, model, device)
