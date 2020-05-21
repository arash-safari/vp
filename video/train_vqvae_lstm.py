from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch


def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

    return zeros.scatter(scatter_dim, y_tensor, 1)


def train(model,input_channel, loader, callback, epoch_num, device, lr, run_num, ):
    image_samples = 10
    writer_path = 'vqvae_videomnist_1_00099_lstm'
    optimizer = get_optimizer(model, lr)
    model = model.to(device)
    model = nn.DataParallel(model)
    criterion = nn.MSELoss()

    writer = SummaryWriter(log_dir='logs/{}_{}'.format(*[writer_path, run_num]))

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
                callback(sample)
                model.train()

            torch.save(model.state_dict(),
                       dir + 'checkpoints/videomnist/vqvae-lstm/{}/{}.pt'.format(*[run_num, str(epoch).zfill(5)]))
