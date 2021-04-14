import logging

import torch
from torch.utils.data import Dataset

from utils import load_h5_data, split_sequence


class DDDLorenzData(Dataset):
    """
    3D Lorenz Dataset

        data_dir:       string      -> directory to load .h5 data from
        lookahead:      int >= 0    -> number of steps to look into the future
        tau:            int >= 1    -> stepsize of history
        k:              int >= 1    -> number of steps to look into the past
        n_groups:       int >= 1    -> number of h5 groups to draw data from (increases data amount)
        dim:            int >= 1    -> number of dimensions to return as data

    """
    def __init__(self, data_dir, lookahead, tau, k, n_groups=1, dim=3):
        all_entire_trajectories = load_h5_data(data_dir)
        for n in range(n_groups):
            entire_trajectory = all_entire_trajectories[n][:, :dim]
            if n == 0:
                self.histories, self.futures = split_sequence(entire_trajectory, lookahead, tau, k)
            else:
                histories, futures = split_sequence(entire_trajectory, lookahead, tau, k)
                self.histories = torch.cat((self.histories, histories), dim=0)
                self.futures = torch.cat((self.futures, futures), dim=0)
        self.n_samples = int(self.histories.size()[0])
        logging.info("\nConstructed 3D Lorenz dataset with: ")
        logging.info("     lookahead: {} | tau: {} | k: {}".format(lookahead, tau, k))
        logging.info("     size of histories: {} | size of futures: {}\n".format(self.histories.size(), self.futures.size()))

    def __getitem__(self, item):
        return self.histories[item], self.futures[item]

    def __len__(self):
        return self.n_samples
