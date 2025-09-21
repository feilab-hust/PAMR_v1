from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.dataset_INR import Dataset_INR
from dataset.utils import DataloaderMode


def create_dataloader(hp_fpm, hp_net, mode, rank, world_size):
    dataset = Dataset_INR(hp_fpm, hp_net, mode)
    train_use_shuffle = False
    sampler = None
    if world_size > 1 and hp_net.data.divide_dataset_per_gpu:
        sampler = DistributedSampler(dataset, world_size, rank)
        train_use_shuffle = False

    if mode is DataloaderMode.train:
        return DataLoader(
            dataset=dataset,
            batch_size=hp_net.train.batch_size,
            shuffle=train_use_shuffle,
            sampler=sampler,
            num_workers=hp_net.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    elif mode is DataloaderMode.test:
        return DataLoader(
            dataset=dataset,
            batch_size=hp_net.test.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=hp_net.test.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        raise ValueError("invalid dataloader mode {}".format(mode))

