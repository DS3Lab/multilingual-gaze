import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class GazeDataLoader(DataLoader):
    def __init__(self, cf, arrays, target_pad, mode):
        input_numpy, target_numpy, mask_numpy = map(list, zip(*arrays))

        self.target_pad = target_pad

        dataset = TensorDataset(torch.as_tensor(input_numpy),
                                torch.as_tensor(target_numpy),
                                torch.as_tensor(mask_numpy))
        sampler = RandomSampler(dataset) if mode == "train" else SequentialSampler(dataset)
        batch_size = cf.train_bs if mode == "train" else cf.eval_bs

        super().__init__(dataset, sampler=sampler, batch_size=batch_size)
