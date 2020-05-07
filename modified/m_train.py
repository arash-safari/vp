from m_trainer import train

dataset_path = 'mnist'
n_run = 0
folder_name = 'vqva'
start_epoch = -1
end_epoch = -1
batch_size = -1
lr = -1
sched=None
device='cuda'
size=28
amp=None
# amp='O0'


train(dataset_path, n_run, folder_name, start_epoch=start_epoch, end_epoch=end_epoch, batch_size=end_epoch, sched=sched, device=device,
          size=size, lr=lr,
          amp=amp)
