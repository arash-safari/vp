import sys
sys.path.append('../image/modified')
import torch
import lmdb
from dataset import VideoMnistDataset
from dataloader import video_mnist_dataloader
from m_extract_vqvae import extract1

def do_extract(model, ckpt_path,lamda_name ,device , batch_size ):
    map_size = 100 * 1024 * 1024 * 1024
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    model.eval()
    env = lmdb.open(lamda_name, map_size=map_size)
    dataset = VideoMnistDataset('datasets/mnist/moving_mnist/mnist_test_seq.npy', 1, 0, 20000)
    loader = video_mnist_dataloader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
    extract1(env, loader, model, device)
