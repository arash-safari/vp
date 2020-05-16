import sys

sys.path.append('../image/modified')
import torch
import lmdb
from tqdm import tqdm
from video.dataset import CodeRowVideoMnist
import pickle


def extract_code(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for frame, video_ind, frame_ind in pbar:
            frame = frame.to(device)

            _, _, _id = model.module.encode(frame)
            _id = _id.detach().cpu().numpy()

            for frame_ind, video_ind, _id in zip(frame_ind, video_ind, _id):
                row = CodeRowVideoMnist(id=_id, frame_ind=frame_ind, video_ind=video_ind)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))


def extract(model, lamda_name, device,loader):
    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(lamda_name, map_size=map_size)
    extract_code(env, loader, model, device)
