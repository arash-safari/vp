from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from video.LSTM_PixelSnail import LSTM_PixelSnail
import numpy as np

def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1).permute(0, 3, 1, 2)


def one_hot_to_int(y):
    y_trans = y.permute(0, 2, 3, 1)
    y_trans = y_trans.argmax(dim=-1)
    return y_trans


def train(lstm_model,cnn_model, pixel_model,input_channel, loader, callback, epoch_num, device, lr, run_num, image_samples=1):
    writer_path = 'vqvae_videomnist_1_00099_lstm_pixelsnail'

    model = LSTM_PixelSnail(lstm_model,cnn_model,pixel_model)
    model = model.to(device)
    model = nn.DataParallel(model)
    criterion = nn.MSELoss()
    optimizer = get_optimizer(model, lr)

    writer = SummaryWriter(log_dir='logs/{}_{}'.format(*[writer_path, run_num]))

    for epoch in range(epoch_num):
        loader = tqdm(loader)
        mse_sum = 0
        mse_n = 0
        for iter, (frames, video_inds, frame_inds) in enumerate(loader):
            model.zero_grad()
            inputs_ = []
            f0 = torch.zeros(frames.shape[0],1,input_channel,frames.shape[2],frames.shape[3])
            f0 = f0.to(device)
            inputs_.append(f0)
            for i in range(frames.shape[1] - 1):
                input_ = _to_one_hot(frames[:, i, :, :], input_channel).float()
                input_ = input_.to(device)

                inputs_.append(input_.unsqueeze(dim=1))

            inputs_ = torch.cat(inputs_, dim=1)
            cells_state = None
            for i in range(frames.shape[1] ):

                pred, cells_state = model(inputs_[:, i:i+2, :, :, :],cells_state)

                loss = criterion(pred, inputs_[:, i+1, :, :, :])
                loss.backward()

                optimizer.step()

                mse_sum += loss.item() * input_.shape[0]
                mse_n += input_.shape[0]
                lr = optimizer.param_groups[0]['lr']
            cells_state.detach()
            if iter % 200 is 0:
                loader.set_description(
                    (
                        'iter: {iter + 1}; mse: {loss.item():.5f}; '
                        f'avg mse: {mse_sum / mse_n:.5f}; '
                        f'lr: {lr:.5f}'
                    )
                )
            if iter % 200 is 0:
                writer.add_scalar('Loss/train', mse_sum / mse_n, epoch_num)
                sample = pred[:image_samples, :, :, :]
                sample = one_hot_to_int(sample)
                callback(sample, frames[:image_samples, -1, :, :].to(device), epoch)

            torch.save(model.state_dict(),
                       '../video/checkpoints/videomnist/vqvae-lstm/{}/{}.pt'.format(*[run_num, str(epoch).zfill(5)]))


