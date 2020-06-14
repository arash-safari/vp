from torch.utils.data import Dataset
import numpy as np
from collections import namedtuple
import lmdb
import pickle
import torch
import os
import cv2
import imageio

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


class Kth_Breakfast_VideoDataset(Dataset):
    def __init__(self, path, frame_len):
        self.frame_len = int(frame_len)
        bpath = path + '/kth_breakfast/'
        self.videos = []
        all_subdirs = os.listdir(bpath)[:-1]
        all_subdirs.sort()
        self.video_index = {}
        self.frame_video_index = {}
        self.index = 0
        for vid_ind, subdir in enumerate(all_subdirs):
            print(vid_ind)
            filepath = bpath + subdir + '/kinect_rgb.mp4'
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                print("Error opening video stream or file")
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            self.video_index[vid_ind] = self.index
            for i in range(len(frames) - frame_len + 1):
                self.frame_video_index[self.index] = vid_ind
                self.index += 1

            self.videos.append(frames)
        cap.release()
        cv2.destroyAllWindows()
    def __len__(self):
        return self.index

    def __getitem__(self, i):
        vid_ind = self.frame_video_index[i]
        frame_ind = self.video_index[vid_ind] - i
        return self.videos[vid_ind][frame_ind: frame_ind + self.frame_len][:, :], vid_ind, frame_ind


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
