from torch.utils.data import Dataset
import numpy as np
from collections import namedtuple
import lmdb
import pickle
import torch
import os
import cv2
import imageio
import h5py

CodeRowVideoMnist = namedtuple('CodeRowVideoMnist', ['ids', 'video_ind'])


class MnistVideoDataset(Dataset):
    def __init__(self, path, frame_len):
        self.frame_len = int(frame_len)
        self.frames = np.load(path)
        self.frames = self.frames.swapaxes(0, 1).astype(np.float32)
        self.frames[self.frames > 0] = 1.
        frames_shape = self.frames.shape
        videos_num = frames_shape[0]
        video_len = frames_shape[1]
        self.sample_per_video = video_len - frame_len + 1
        self.length = videos_num * self.sample_per_video

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        video_ind = int(index / self.sample_per_video)
        frame_ind = index - video_ind * self.sample_per_video

        return self.frames[video_ind, frame_ind: frame_ind + self.frame_len, :, :], video_ind, frame_ind


# class MnistVideoDataset2(Dataset):
#     def __init__(self, path, frame_len):
#         self.frame_len = int(frame_len)
#         self.frames = np.load(path)
#         self.frames = self.frames.swapaxes(0, 1).astype(np.float32)
#         self.frames[self.frames > 0] = 1.
#         frames_shape = self.frames.shape
#         videos_num = frames_shape[0]
#         video_len = frames_shape[1]
#         self.sample_per_video = video_len - frame_len + 1
#         self.length = (videos_num * self.sample_per_video * (self.sample_per_video -1) )/2
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, index):
#         video_ind = int(2 * index / (self.sample_per_video*(self.sample_per_video - 1)))
#         frame_ind = index - video_ind * (self.sample_per_video*(self.sample_per_video - 1))
#         return self.frames[video_ind, frame_ind: min(frame_ind + self.frame_len, self.frames.shape[1]), :, :], video_ind, frame_ind

class lmdb_video(Dataset):
    def __init__(self, env_path):
        self.env = lmdb.open(
            env_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', env_path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            self.videos_ind = pickle.loads(txn.get('videos_ind'.encode('utf-8')).decode('utf-8'))
            self.frames_ind = pickle.loads(txn.get('frames_ind'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            video_ind = int(index / self.sample_per_video)
            frame_ind = index - video_ind * self.sample_per_video

            key = str(video_ind).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.ids[frame_ind: frame_ind + self.frame_len]), row.video_ind, frame_ind


class MnistVideoCodeLMDBDataset(Dataset):
    def __init__(self, path, frame_len):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.frame_len = int(frame_len)
        video_len = 20
        self.sample_per_video = video_len - frame_len + 1

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8')) * self.sample_per_video

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            video_ind = int(index / self.sample_per_video)
            frame_ind = index - video_ind * self.sample_per_video

            key = str(video_ind).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.ids[frame_ind: frame_ind + self.frame_len]), row.video_ind, frame_ind


class MnistVideoCodeLMDBDataset2(Dataset):
    def __init__(self, path, frame_len):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.frame_len = int(frame_len)
        video_len = 20
        self.sample_per_video = video_len - frame_len + 1

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8')) * int(
                (self.sample_per_video * (self.sample_per_video + 1)) / 2)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            video_ind = int(index / self.sample_per_video)
            frame_ind = index - video_ind * int((self.sample_per_video * (self.sample_per_video + 1)) / 2)

            key = str(video_ind).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(
            row.ids[frame_ind: min(frame_ind + self.frame_len, row.ids.shape[0])]), row.video_ind, frame_ind
