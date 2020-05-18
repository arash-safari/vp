from torch import nn, optim
from video.LSTM_VideoMnist import LSTM
from video.dataset import MnistVideoCodeLMDBDataset
from video.dataloader import video_mnist_dataloader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import utils
import torch
import numpy as np


def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)


def train(vqvae_model):
    lambda_name = 'vqvae_videomnist_1_00099'
    input_channel = 8
    epoch_num = 10
    batch_size = 64
    device = 'cuda'
    lr = 0.0001
    run_num = 1
    image_samples = 10

    dataset = MnistVideoCodeLMDBDataset(lambda_name, 3)
    loader = video_mnist_dataloader(dataset, batch_size, shuffle=True)
    model = LSTM(2, input_channel, 32, 3)
    videomnist_path = '../video/datasets/mnist/moving_mnist/mnist_test_seq.npy'
    orginal_frames = np.load(videomnist_path)
    orginal_frames = orginal_frames.swapaxes(0, 1).astype(np.float32)
    orginal_frames[orginal_frames > 0] = 1.

    optimizer = get_optimizer(model, lr)
    model = model.to(device)
    model = nn.DataParallel(model)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir='logs/{}_{}'.format(*['vqvae_videomnist_1_00099_lstm', run_num]))

    for epoch in range(epoch_num):
        loader = tqdm(loader)
        mse_sum = 0
        mse_n = 0
        for iter, (frames, video_inds, frame_inds) in enumerate(loader):
            model.zero_grad()
            for i in range(input.shape[1] - 1):
                input = _to_one_hot(frames[:, i, :, :], input_channel)
                output = _to_one_hot(frames[:, i + 1, :, :], input_channel)
                pred = model(input)
                loss = criterion(pred, output)
                loss.backward()
                optimizer.step()
                mse_sum += loss.item() * input.shape[0]
                mse_n += input.shape[0]

            lr = optimizer.param_groups[0]['lr']
            if iter % 200 is 0:
                loader.set_description(
                    (
                        'iter: {iter + 1}; mse: {recon_loss.item():.5f}; '
                        f'avg mse: {mse_sum / mse_n:.5f}; '
                        f'lr: {lr:.5f}'
                    )
                )
            if iter is 0 and epoch > 0:
                writer.add_scalar('Loss/train', mse_sum / mse_n, epoch_num)
                model.eval()
                sample = pred[0, :image_samples, :, :]
                o_frames = orginal_frames[video_inds[0], :image_samples, :, :]
                with torch.no_grad():
                    sample = vqvae_model.module.decode_code(sample)

                    utils.save_image(
                        torch.cat([sample, o_frames], 0),
                        dir + 'samples/videomnist/vqvae/{}/{}.png'.format(*[run_num, epoch]),
                        nrow=image_samples,
                        normalize=True,
                        range=(-1, 1),
                    )
                    model.train()

            torch.save(model.state_dict(),
                       dir + 'checkpoints/videomnist/vqvae-lstm/{}/{}.pt'.format(*[run_num, str(epoch).zfill(5)]))
