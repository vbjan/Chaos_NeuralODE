import torch
from torch.utils.data import Dataset
from utils import load_h5_data, split_sequence, make_batches_from_stack


class DDDLorenzData(Dataset):
    def __init__(self, data_dir, lookahead, tau, k):
        all_entire_trajectories = load_h5_data(data_dir)
        entire_trajectory = all_entire_trajectories[0][:,:]
        self.histories, self.futures = split_sequence(entire_trajectory, lookahead, tau, k)
        print("histories . {}, futurers: {}".format(self.histories.size(), self.futures.size()))
        self.n_samples = int(self.histories.size()[0])
        print("data len: {}".format(self.n_samples))

    def __getitem__(self, item):
        return self.histories[item], self.futures[item]

    def __len__(self):
        return self.n_samples
