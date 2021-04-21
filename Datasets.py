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
        max_len:        int >= 1    -> length of trajectory to generate data from
        normalize:      bool        -> specifies wether to normalize data

    """
    def __init__(self, data_dir, lookahead, tau, k, max_len, n_groups=1, dim=3, normalize=False):
        self.data_dir = data_dir
        self.lookahead = lookahead
        self.tau = tau
        self.k = k
        self.max_len = max_len
        self.data_dim = dim

        all_entire_trajectories = load_h5_data(data_dir)
        for n in range(n_groups):
            entire_trajectory = all_entire_trajectories[n][:, :dim]

            # shorten trajectory if less data is wanted
            if len(entire_trajectory) > self.max_len:
                entire_trajectory = entire_trajectory[:self.max_len]
            else:
                logging.warning("max_len = {} exceeds maximum length of dataset trajectory = {}".format(self.max_len, len(entire_trajectory)))
                pass

            # split trajectory and write it into futures and histories
            if n == 0:
                self.histories, self.futures = split_sequence(entire_trajectory, self.lookahead, self.tau, self.k)
            else:
                histories, futures = split_sequence(entire_trajectory, self.lookahead, self.tau, self.k)
                self.histories = torch.cat((self.histories, histories), dim=0)
                self.futures = torch.cat((self.futures, futures), dim=0)

        # normalize the data w.r.t. to the maximum value
        dims = len(self.histories[0, 0, :])
        for dim in range(dims):
            max_value = torch.max(self.histories[:, :,  dim])
            if normalize:
                self.histories[:, :, dim] = self.histories[:, :, dim]/max_value
                self.futures[:, :, dim] = self.futures[:, :, dim]/max_value

        self.n_samples = int(self.histories.size()[0])
        logging.info("\nConstructed 3D Lorenz dataset with: ")
        logging.info("     lookahead: {} | tau: {} | k: {}".format(self.lookahead, self.tau, self.k))
        logging.info("     size of histories: {} | size of futures: {}\n".format(self.histories.size(), self.futures.size()))

    def __getitem__(self, item):
        return self.histories[item], self.futures[item]

    def __len__(self):
        return self.n_samples


class VanDePol(Dataset):
    """
        Van de Pol Oscilator dataset:
            data_dir:       string      -> directory to load .h5 data from
            lookahead:      int >= 0    -> number of steps to look into the future
            tau:            int >= 1    -> stepsize of history
            k:              int >= 1    -> number of steps to look into the past
            max_len:        int >= 1    -> length of trajectory to generate data from
            normalize:      bool        -> specifies wether to normalize data
    """
    def __init__(self, data_dir, lookahead, tau, k, max_len, normalize=False):
        super(VanDePol, self).__init__()
        self.data_dir = data_dir
        self.lookahead = lookahead
        self.tau = tau
        self.k = k
        self.max_len = max_len
        self.data_dim = 2

        # load simulated data
        self.traj = torch.load(self.data_dir)

        # shorten trajectory if less data is wanted
        if len(self.traj) > self.max_len:
            self.traj = self.traj[:self.max_len]
        else:
            logging.warning("max_len = {} exceeds maximum length of dataset trajectory = {}".format(self.max_len, len(
                self.traj)))
            pass

        # split the trajectory
        self. histories, self.futures = split_sequence(self.traj, self.lookahead, self.tau, self.k)

        # normalize data if specified
        if normalize:
            dims = len(self.histories[0, 0, :])
            for dim in range(dims):
                max_value = torch.max(self.histories[:, :, dim])
                if normalize:
                    self.histories[:, :, dim] = self.histories[:, :, dim] / max_value
                    self.futures[:, :, dim] = self.futures[:, :, dim] / max_value

        self.n_samples = int(self.histories.size()[0])
        logging.info("\nConstructed Van de Pol dataset with: ")
        logging.info("     lookahead: {} | tau: {} | k: {}".format(self.lookahead, self.tau, self.k))
        logging.info("     size of histories: {} | size of futures: {}\n".format(self.histories.size(), self.futures.size()))

    def __getitem__(self, item):
        return self.histories[item], self.futures[item]

    def __len__(self):
        return self.n_samples