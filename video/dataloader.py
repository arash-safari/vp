from torch.utils.data import DataLoader


def video_mnist_dataloader(dataset, batch_size, shuffle=True, num_workers=4, drop_last=True):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last
    )
    return loader
