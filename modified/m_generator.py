from m_sample import make_sample
from m_util import load_model 
 
dataset = 'mnist'
n_run = 0 
vqvae_epoch = 30
top_epoch = 10
bottom_epoch = 20

vqvae, top ,bottom , middle, sample_dir = load_model(device = 'cuda', dataset = dataset, n_run=n_run,
 vqvae_epoch = vqvae_epoch, top_epoch = top_epoch, bottom_epoch = bottom_epoch, middle_epoch=-1)

make_sample(vqvae, top ,bottom , middle, sample_dir, '0.png', temp=1.0)

