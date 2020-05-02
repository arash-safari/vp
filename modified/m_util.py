import sys
sys.path.append('../')
import torch
from vqvae import VQVAE
from m_conf_parser import pixel_object

def get_path(dataset, n_run, model, file_type, checkpoint=0):
    file_path =  '../checkpoint/{}/{}/'.format(*[dataset, n_run])
    
    if checkpoint < 10:
        checkpoint = '00{}'.format(checkpoint)
    elif checkpoint < 100:
        checkpoint = '0{}'.format(checkpoint)
    if  model == 'vqvae':
        file_path += 'vqvae/'
    elif model == 'top':
        file_path += 'pixelsnail/top/'
    elif model == 'bottom':
        file_path += 'pixelsnail/bottom/'
    elif model == 'middle':
        file_path += 'pixelsnail/middle/'
        
    if file_type == 'conf':
        file_path += 'conf.ini'
    else:
        if  model == 'vqvae':
            ckpt = 'vqvae_{}.pt'.format(checkpoint)
        elif model == 'top':
           ckpt = 'pixelsnail_top_{}.pt'.format(checkpoint)
        elif model == 'bottom':
            ckpt = 'pixelsnail_bottom_{}.pt'.format(checkpoint)
        elif model == 'middle':
            ckpt = 'pixelsnail_middle_{}.pt'.format(checkpoint)
        file_path += ckpt
    return file_path

def get_sample_dir(dataset, n_run ):
    return '../checkpoint/{}/{}/sample'.format(*[dataset, n_run])

def load_part(model, checkpoint, device):
    ckpt = torch.load(checkpoint)
    if 'model' in ckpt:
        ckpt = ckpt['model']
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model

def load_model(device, dataset, n_run, vqvae_epoch, top_epoch, bottom_epoch , middle_epoch=-1):

    top_conf_path = get_path(dataset, n_run, 'top', 'conf', checkpoint=-1)
    middle_conf_path = get_path(dataset, n_run, 'middle', 'conf', checkpoint=-1)
    bottom_conf_path = get_path(dataset, n_run, 'bottom', 'conf', checkpoint=-1)

    top_checkpoint_path = get_path(dataset, n_run, 'top', 'check_point', checkpoint=top_epoch)
    middle_checkpoint_path = get_path(dataset, n_run, 'middle', 'check_point', checkpoint=middle_epoch)
    bottom_checkpoint_path = get_path(dataset, n_run, 'bottom', 'check_point', checkpoint=bottom_epoch)

    vqvae_checkpoint_path = get_path(dataset, n_run, 'vqvae', 'check_point', checkpoint=vqvae_epoch)

    vqvae = VQVAE()
    top = pixel_object(top_conf_path) 
    bottom = pixel_object(bottom_conf_path)
    if middle_epoch > 0:
        middle = pixel_object(middle_conf_path)

    model_vqvae = load_part(vqvae , vqvae_checkpoint_path, device)
    model_top = load_part(top, top_checkpoint_path, device)
    model_bottom = load_part(bottom, bottom_checkpoint_path, device)  
    model_middle = None  
    if middle_epoch > 0:
        model_middle = load_part(middle, middle_checkpoint_path, device)
    sample_dir = get_sample_dir(dataset, n_run )
    return (model_vqvae, model_top, model_bottom, model_middle, sample_dir)
     
