import sys
sys.path.append('../')
import torch
from vqvae import VQVAE
from pixelsnail import PixelSNAIL
from modified.m_conf_parser import model_option_parser, training_params_parser


def get_sample_dir(dataset, n_run ):
    return '..\\checkpoint\\{}\\{}\\sample'.format(*[dataset, n_run])

def load_part(model, checkpoint, device):

    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    return model

def create_model_object (model_type, options):
    if model_type == 'pixelsnail':
        return PixelSNAIL(
            shape = options['shape'],
            n_class = options['n_class'],
            channel = options['channel'],
            kernel_size = options['kernel_size'],
            n_block = options['n_block'],
            n_res_block = options['n_res_block'],
            res_channel = options['res_channel'],
            dropout = options['dropout'],
            n_cond_res_block = options['n_cond_res_block'],
            cond_res_channel = options['cond_res_channel'],
            cond_res_kernel = options['cond_res_kernel'],
            n_out_res_block = options['n_out_res_block'],
            attention = options['attention']
        )
    elif model_type == 'vqvae':
        return VQVAE()

def get_model_type(folder_name):
    if folder_name in ['top','bottom','middle']:
        return 'pixelsnail'
    elif folder_name == 'vqvae':
        return 'vqvae'

def get_path(dataset, n_run, model, file_type, checkpoint=0):
    file_path =  '..\\checkpoint\\{}\\{}\\'.format(*[dataset, n_run])
    model_type = get_model_type(model)
    if  model_type == 'vqvae':
        file_path += 'vqvae\\'
    elif model_type == 'pixelsnail':
        file_path += 'pixelsnail\\{}\\'.format(model)

    if file_type == 'conf':
        file_path += 'conf.ini'
    elif model == 'vqvae':
            ckpt = '{}_{}.pt'.format(*[model, checkpoint])
            file_path += ckpt
    return file_path

def conf_parser(dataset, n_run, folder_name):
    conf_path = get_path(dataset, n_run, folder_name, 'conf')
    model_type = get_model_type(folder_name)
    options = model_option_parser(model_type, conf_path)
    train_params = training_params_parser(conf_path)
    return options, train_params

def model_object_parser(dataset, n_run, folder_name):
    model_type = get_model_type(folder_name)
    options,_ = conf_parser(dataset, n_run, folder_name)
    return create_model_object(model_type, options)
    
    
def load_model(device, dataset, n_run, vqvae_epoch, top_epoch, bottom_epoch , middle_epoch=-1):
    
    top_checkpoint_path = get_path(dataset, n_run, 'top', 'check_point', checkpoint=top_epoch)
    middle_checkpoint_path = get_path(dataset, n_run, 'middle', 'check_point', checkpoint=middle_epoch)
    bottom_checkpoint_path = get_path(dataset, n_run, 'bottom', 'check_point', checkpoint=bottom_epoch)
    vqvae_checkpoint_path = get_path(dataset, n_run, 'vqvae', 'check_point', checkpoint=top_epoch)
    
    vqvae_obj = model_object_parser(dataset, n_run, 'vqvae')
    top_obj = model_object_parser(dataset, n_run, 'top')
    bottom_obj = model_object_parser(dataset, n_run, 'bottom')
    if middle_epoch > 0:
        middle_obj = model_object_parser(dataset, n_run, 'middle')
    
    model_vqvae = load_part(vqvae_obj , vqvae_checkpoint_path, device)
    model_top = load_part(top_obj, top_checkpoint_path, device)
    model_bottom = load_part(bottom_obj, bottom_checkpoint_path, device)    
    if middle_epoch > 0:
        model_middle = load_part(middle_obj, middle_checkpoint_path, device)
    
    sample_dir = get_sample_dir(dataset, n_run )
    return (model_vqvae, model_top, model_bottom, model_middle, sample_dir)
         
    