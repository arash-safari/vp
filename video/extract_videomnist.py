import sys

sys.path.append('../image/modified')
import torch
import lmdb
from dataset import VideoMnistDataset
from dataloader import video_mnist_dataloader
from m_extract_vqvae import extract1
from m_vqvae import VQVAE_1


def do_extract(model, ckpt_path, lamda_name, device, loader):
    map_size = 100 * 1024 * 1024 * 1024
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    model.eval()
    env = lmdb.open(lamda_name, map_size=map_size)
    extract1(env, loader, model, device)


batch_size = 100
device = 'cuda'
dataset = VideoMnistDataset('datasets/mnist/moving_mnist/mnist_test_seq.npy', 1, 0, 20000)
loader = video_mnist_dataloader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
lamda_name = 'vqvae_videomnist_1'
ckpt_path = './checkpoints/videomnist/vqvae/1/1.pt'
model = VQVAE_1(in_channel=1,
                channel=32,
                n_res_block=2,
                n_res_channel=16,
                embed_dim=8,
                n_embed=4,
                decay=0.99, )
