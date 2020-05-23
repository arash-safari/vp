import sys

sys.path.append('../image/modified')
import torch
import lmdb
from tqdm import tqdm
from video.dataset import CodeRowVideoMnist,MnistVideoDataset, MnistVideoCodeLMDBDataset
import pickle

from video.dataloader import video_mnist_dataloader


def extract_code(lmdb_env, loader, model, device):
    print('extract code ')
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)
        print('lmdb env')
        for frames_batch, video_ind_batch, frame_ind_batch in pbar:
            print('epoch {}'.format(index))
            for i,frames in enumerate(frames_batch):
                frames = frames.to(device)
                frames = frames.unsqueeze(1)
                _, _, _ids = model.module.encode(frames)
                _ids = _ids.detach().cpu().numpy()
                row = CodeRowVideoMnist(ids=_ids,  video_ind=video_ind_batch[i])
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


def extract(model, lamda_name, device, video_batch):
    dataset_video = MnistVideoDataset(path='../video/datasets/mnist/moving_mnist/mnist_test_seq.npy', frame_len=20)
    loader = video_mnist_dataloader(dataset_video, video_batch, shuffle=False, num_workers=4, drop_last=True)

    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(lamda_name, map_size=map_size)

    extract_code(env, loader, model, device)
