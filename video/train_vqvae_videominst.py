import sys
sys.path.append('../image/modified')
from m_vqvae import VQVAE_1
from torch import optim, nn
import torch
from torchvision import utils
from dataset import VideoMnistDataset
from dataloader import video_mnist_dataloader
def get_optimizer(model, lr):
    return optim.Adam(model.parameters(), lr=lr)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

lr = 0.001
device = 'cuda'
epoch_num = 100
batch_size = 100
run_num = 1
image_samples = 10
model = VQVAE_1(in_channel=1,
            channel=32,
            n_res_block=2,
            n_res_channel=16,
            embed_dim=16,
            n_embed=4,
            decay=0.99,)
dataset = VideoMnistDataset('datasets/mnist/moving_mnist/mnist_test_seq.npy', 1, 0,20000)
loader  = video_mnist_dataloader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=True)
optimizer = get_optimizer(model, lr)
model = model.to(device)
model = nn.DataParallel(model)
criterion = nn.MSELoss()
writer = SummaryWriter(log_dir='logs/{}_{}'.format(*['videomnist-vqvae', run_num]))


for epoch in range(epoch_num):
    latent_loss_weight = 0.25
    mse_sum = 0
    mse_n = 0
    loader = tqdm(loader)

    for iter, img in enumerate(loader):
        model.zero_grad()
        img = img[:,0:1,:,:]
        img = img.to(device)
        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']
        if iter % 200 is 0:
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
            sample = img[:image_samples]

            with torch.no_grad():
                out, _ = model(sample)
            # save samples to see result

            out = (out > 0.5).float()
            utils.save_image(
                torch.cat([sample, out], 0),
                'samples/videomnsit/vqvae/{}/{}.png'.format(*[run_num,epoch_num]),
                nrow=image_samples,
                normalize=True,
                range=(-1, 1),
            )
            model.train()

    torch.save(model.state_dict(), 'checkpoints/videomnist/vqvae/{}/{}.pt'.format(*[run_num, str(epoch).zfill(5)]))
