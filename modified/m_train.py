from m_trainer import train
from m_data_loader import get_lmdb_pixel_loader, get_image_loader
from m_preprocessing import get_transform

dataset_name = 'mnist'
n_run = 0
folder_name = 'vqvae'
start_epoch = -1
end_epoch = -1
batch_size = -1
lr = -1
sched=None
device='cuda'
size=28
amp=None
# amp='O0'

if folder_name == 'top':
    loader = get_lmdb_pixel_loader(dataset_name, n_run, batch_size,
                                   x_name='top', cond=None, shuffle=True, num_workers=4)
elif folder_name == 'bottom':
    loader = get_lmdb_pixel_loader(dataset_name, n_run, batch_size,
                                   x_name='bottom', cond='top', shuffle=True, num_workers=4)
elif folder_name == 'vqvae':
    loader = get_image_loader(dataset_name, batch_size, transform= get_transform(size), shuffle=True, num_workers=4)

train(
       folder_name,
       loader,
       dataset_name,
       n_run,
       start_epoch=start_epoch,
       end_epoch=end_epoch,
       batch_size=batch_size,
       sched=sched,
       device=device,
       size=size,
       lr=lr,
       amp=amp,

       )
