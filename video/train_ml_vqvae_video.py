import sys

sys.path.append('../image/modified')
from m_vqvae_multi_level import VQVAE_ML
from torch import optim, nn
import torch
from torchvision import utils
from dataset import lmdb_video
from dataloader import video_loader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)


def train(dataset, model, epoch_num, batch_size, lr, device, run_num, image_samples, callback, loss_func):
    loader = video_loader(dataset, batch_size, shuffle=False, num_workers=1, drop_last=True)
    optimizer = get_optimizer(model, lr)
    model = model.to(device)
    # model = nn.DataParallel(model)
    if loss_func=="mse":
        criterion = nn.MSELoss()
    elif loss_func=="l1":
        criterion = nn.L1Loss()
    writer = SummaryWriter(log_dir='logs/{}_{}'.format(*['kth-breakfast-vqvae', run_num]))

    for epoch in range(epoch_num):
        latent_loss_weight = 0.25
        mse_sum = 0
        mse_n = 0
        loader = tqdm(loader)

        for iter, frame in enumerate(loader):
            # for img in video:
            # video = video.permute(0,3,1,2)
            model.zero_grad()
            frame = frame.float().to(device)
            out, latent_loss = model(frame)
            recon_loss = criterion(out, frame)
            latent_loss = latent_loss.mean()
            loss = recon_loss + latent_loss_weight * latent_loss
            loss.backward()

            optimizer.step()

            mse_sum += recon_loss.item() * frame.shape[0]
            mse_n += frame.shape[0]

            lr = optimizer.param_groups[0]['lr']
            loader.set_description(
                (
                    'iter: {iter + 1}; mse: {recon_loss.item():.5f}; '
                    f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                    f'lr: {lr:.5f}'
                )
            )
            if iter is 0 and epoch > 0:
                writer.add_scalar('Loss/train', mse_sum / mse_n, epoch_num)
                model.eval()
                sample = frame[:image_samples]

                with torch.no_grad():
                    out, _ = model(sample)
                # save samples to see result

                # out = (out > 0.5).float()
                callback(sample, out, epoch)

                model.train()

            torch.save(model.state_dict(),
                        '../video/checkpoints/kth-breakfast/vqvae/{}/{}.pt'.format(*[run_num, str(epoch).zfill(5)]))
