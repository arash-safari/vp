from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from video.LSTM_PixelSnail3 import LSTM_PixelSnail3
import numpy as np


def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1).permute(0, 3, 1, 2)


def add_noise(frames, noise, max_val):
    size = frames.size()
    numchange = int(noise * size[2] * size[3])

    rand = np.random.randint(max_val, size=(size[0], size[1], numchange))
    ex = np.ones((size[0], size[1], size[2] * size[3] - numchange)) * -1
    rand = np.concatenate((rand, ex), axis=2)
    for i in range(size[0]):
        for j in range(size[1]):
            np.random.shuffle(rand[i, j, :])
    rand = rand.reshape((size[0], size[1], size[2], size[3]))
    rand = torch.from_numpy(rand).long()
    rand[rand < 0] = frames[rand < 0]
    return rand


def one_hot_to_int(y):
    y_trans = y.permute(0, 2, 3, 1)
    y_trans = y_trans.argmax(dim=-1)
    return y_trans


def train(chekpoint, lstm_model, pixel_model, input_channel, loader, callback, epoch_num, device, lr,
          run_num, num_frame_learn=8, image_samples=1, noise=0.05, max_val=16):
    writer_path = 'vqvae_videomnist_{}_00099_lstm_pixelsnail'.format(run_num)

    model = LSTM_PixelSnail3(lstm_model, pixel_model)
    model = nn.DataParallel(model)
    model = model.to(device)

    if chekpoint > 0:
        ckpt_path = "../video/checkpoints/videomnist/vqvae-lstm-pixelsnail/{}/{}.pt".format(run_num,
                                                                                            str(chekpoint).zfill(5))
        model.load_state_dict(torch.load(ckpt_path))

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr)

    writer = SummaryWriter(log_dir='logs/{}_{}'.format(*[writer_path, run_num]))
    epoch_start = 0
    if chekpoint > 0:
        epoch_start = chekpoint + 1
    for epoch in range(epoch_start, epoch_num + epoch_start):
        loader = tqdm(loader)
        mse_sum = 0
        mse_n = 0
        for iter, (frames, video_inds, frame_inds) in enumerate(loader):
            inputs_ = []
            noisi_frames = add_noise(frames, noise, max_val)
            f0 = torch.zeros(frames.shape[0], 1, input_channel, frames.shape[2], frames.shape[3])
            f0 = f0.to(device)
            inputs_.append(f0)
            for i in range(frames.shape[1]):
                input_ = _to_one_hot(noisi_frames[:, i, :, :], input_channel).float()
                input_ = input_.to(device)

                inputs_.append(input_.unsqueeze(dim=1))

            inputs_ = torch.cat(inputs_, dim=1)
            cells_state = None
            loss = 0
            model.zero_grad()
            frames = frames.to(device)
            for i in range(num_frame_learn):
                pred, cells_state = model(torch.cat([inputs_[:, i:i + 2, :, :, :],
                                                     inputs_[:, num_frame_learn + 1, :, :, :].unsqueeze(dim=1)
                                                     ], dim=1), cells_state)
                loss += criterion(pred, frames[:, i, :, :])

            preds = inputs_[:, num_frame_learn + 1, :, :, :].unsqueeze(dim=1)
            num_frame_preds = frames.shape[1] - num_frame_learn - 1

            for i in range(num_frame_preds):
                model_input = torch.cat([preds[:, -1:, :, :, :],
                                         inputs_[:, num_frame_learn + i + 2, :, :, :].unsqueeze(dim=1),
                                         inputs_[:, num_frame_learn + 1, :, :, :].unsqueeze(dim=1)], dim=1)
                pred, cells_state = model(model_input, cells_state)
                preds = torch.cat([preds, pred.unsqueeze(dim=1)], dim=1)
                loss += criterion(pred, frames[:, i + num_frame_learn + 1, :, :])

            loss.backward()
            optimizer.step()

            mse_sum += loss.item() * input_.shape[0]
            mse_n += input_.shape[0]
            lr = optimizer.param_groups[0]['lr']
            loader.set_description(
                (
                    'iter: {iter + 1}; mse: {loss.item():.5f}; '
                    f'avg mse: {mse_sum / mse_n:.5f}; '
                    f'lr: {lr:.5f}'
                )
            )

            if iter % 200 is 0:
                writer.add_scalar('Loss/train', mse_sum / mse_n, epoch_num)
                samples = None
                for i in range(num_frame_preds):
                    sample = one_hot_to_int(preds[:, i + 1, :, :, :])
                    if samples == None:
                        samples = sample.unsqueeze(dim=1)
                    else:
                        samples = torch.cat([samples, sample.unsqueeze(dim=1)], dim=1)
                callback(samples, frames[:, -(frames.shape[1] - num_frame_learn) + 1:, :, :].to(device), epoch, iter)

            torch.save(model.state_dict(),
                       '../video/checkpoints/videomnist/vqvae-lstm-pixelsnail/{}/{}.pt'.format(
                           *[run_num, str(epoch).zfill(5)]))
