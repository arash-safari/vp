from torch.utils.data import Dataset
import numpy as np
from collections import namedtuple
import lmdb
import pickle
import torch

CodeRowVideoMnist = namedtuple('CodeRowVideoMnist', ['id', 'video_ind', 'frame_ind'])


class VideoMnistDataset(Dataset):
    def __init__(self, path, frame_len, start_sample, end_sample):
        self.frame_len = int(frame_len)
        self.frames = np.load(path)
        self.frames = self.frames.swapaxes(0, 1)[start_sample:end_sample, :, :, :].astype(np.float32)
        self.frames[self.frames > 0] = 1.
        frames_shape = self.frames.shape
        videos_num = frames_shape[0]
        video_len = frames_shape[1]
        # self.frames = self.frames - np.ones(frames_shape) / 2
        # self.frames = 2 * self.frames
        self.sample_per_video = video_len - frame_len + 1
        self.length = videos_num * self.sample_per_video

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_ind = int(index / self.sample_per_video)
        frame_ind = index - video_ind * self.sample_per_video

        return self.frames[video_ind, frame_ind: frame_ind + self.frame_len, :, :], video_ind, frame_ind


class VideoMnistLMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.id), torch.from_numpy(row.video_ind), row.frame_ind
