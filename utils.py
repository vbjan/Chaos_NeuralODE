'''
TODO:
    - write save models and save optimizer function
'''
import torch.nn as nn
import torch
import logging
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


def load_h5_data(directory):
    """
        Loads h5 files containing the Lorenz trajectories
    """
    orig_trajs = []
    with h5py.File(directory, "r") as h5:
        keys = list(h5.keys())
        for key in keys:
            g = h5.get(key)
            data = np.array(g.get("data"))
            orig_trajs.append(data)
    orig_trajs = np.stack(orig_trajs)
    logging.debug("Successfully loaded data from {} directory and stored it in np.stack."
          " size of stack: {}".format(directory, orig_trajs.shape))
    return orig_trajs


class CreationRNN(nn.Module):
    """
    GRU class for encoding and decoding
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, nbatch):
        super(CreationRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.nbatch = nbatch
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.o2o = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.nbatch, self.hidden_dim)  # used to set hidden vector to zeros


    def forward(self, x, hid):
        #skip = x
        x, hid = self.rnn(x, hid)
        return  self.o2o(x), hid


def get_lr(opt):
    """
    :param opt: pytorch optimizer
    :return: learning rate of optimizer
    """
    for param_group in opt.param_groups:
        return param_group['lr']


def save_optimizer(save_dir, opt):
    """
    :param save_dir: location to save in | example: Models/examplemodel
    :param opt: optimizer to be saved (max=1)
    """
    if save_dir is not None:
        ckpt_path = save_dir + 'opt.pth'
        torch.save({
            'optimizer': opt.state_dict(),
        }, ckpt_path)
        logging.debug('Saved optimizer')


def save_models(save_dir, *models):
    """
    :param save_dir: location to save in | example: Models/examplemodel
    :param models:
    :return: any number of pytorch NNs to be saved
    """
    if save_dir is not None:
        i = 0
        for model in models:
            ckpt_path = save_dir + 'model' + str(i) + '.pth'
            torch.save({
                'model' + str(i): model.state_dict()
            }, ckpt_path)
            i += 1
        logging.info('Stored ckpt at {}'.format(ckpt_path))

'''

def save_models(save_dir, opt, *models):
    """
    :param save_dir: location to save in | example: Models/examplemodel
    :param opt: optimizer to be saved (max=1)
    :param models: any number of pytorch NNs to be saved
    """
    if save_dir is not None:
        ckpt_path = save_dir + 'opt.pth'
        torch.save({
            'optimizer': opt.state_dict(),
        }, ckpt_path)
        logging.debug('Saved optimizer')
        i = 0
        for model in models:
            ckpt_path = save_dir + 'model' + str(i) + '.pth'
            torch.save({
                'model'+str(i): model.state_dict()
            }, ckpt_path)
            i += 1
        logging.info('Stored ckpt at {}'.format(ckpt_path))


def load_models(save_dir, opt, *models):
    """
    :param save_dir: location to load from | example: Models/examplemodel
    :param opt: optimizer that was saved with "save_models"
    :param models: the same models in same order as saved with "save_models" to be loaded
    """
    ckpt_path = save_dir + 'opt.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        opt.load_state_dict(checkpoint['optimizer'])
        logging.debug('Loaded optimizer')
        i = 0
        for model in models:
            ckpt_path = save_dir + 'model' + str(i) + '.pth'
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model'+str(i)])
            i += 1
        logging.debug('Loaded all the models')
    logging.debug('Loaded ckpt from {}'.format(ckpt_path))'''


def load_models(save_dir, *models):
    """
    :param save_dir: location to load from | example: Models/examplemodel
    :param models: the same models in same order as saved with "save_models" to be loaded
    """
    i = 0
    for model in models:
        ckpt_path = save_dir + 'model' + str(i) + '.pth'
        print(ckpt_path)
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model' + str(i)])
        i += 1
    logging.debug('Loaded all the models')


def load_optimizer(save_dir, opt):
    """
    :param save_dir: location to load from | example: Models/examplemodel
    :param opt: optimizer that was saved with "save_optimizer"
    """
    ckpt_path = save_dir + 'opt.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        opt.load_state_dict(checkpoint['optimizer'])
        logging.debug('Loaded optimizer')


def split_sequence(sequence, lookahead, tau, k, max_blocks):
    """
    :param sequence: data sequence with dimensions (sequence length, data dimension)
    :param lookahead: number of steps to predict
    :param tau: stepsize of history (see Takens theorem)
    :param k: number of steps of history
    :param max_blocks: maximum number of training points per batch
    :return: torch.tensor of history and future of dimensions (nblocks, future/history length, data dimension)
    """
    sequence_len = sequence.shape[0]
    histories = []
    futures = []
    for i in range((k-1) * tau, sequence_len - lookahead):
        prev_i = i
        histories.append(sequence[i - (k-1) * tau:i + 1:tau])
        futures.append(sequence[i:i + lookahead + 1 :1])
        if len(histories) == max_blocks:
            break
    histories, futures = np.stack(histories), np.stack(futures)
    if histories.shape[0] != max_blocks:
        logging.warning("histories.shape: {}, max_blocks: {}".format(histories.shape, max_blocks))
        assert histories.shape[0] == max_blocks
    histories, futures = torch.from_numpy(histories).float(), torch.from_numpy(futures).float()
    return histories, futures


def make_batches_from_stack(stack, lookahead, tau, k, n_batches, max_blocks=512):
    """
    :param stack: data stack with all the simulated trajectories
    :param lookahead: number of steps to predict
    :param tau: stepsize of history (see Takens theorem)
    :param k: number of steps of history
    :param n_batches: number of batches
    :param max_blocks: maximum number of training points per batch
    :return: lists containing the n_batches amount of histories and futures
    """
    batches_histories = []
    batches_futures = []
    for i in range(n_batches):
        sequence = stack[i][:, :]
        histories, futures = split_sequence(sequence, lookahead, tau, k, max_blocks)
        batches_histories.append(histories)
        batches_futures.append(futures)
    return batches_histories, batches_futures


def z1test(x, show_warnings=True, plotting=False):
    """
    :param x: 1-dimensional sequence of data
    :param show_warnings: alert if any possible issues arising from oversampling
    :param plotting: bool that decides if plots are shown
    :return: K-value that indicates chaos of sequence x. K=0 -> x likely not chaotic and K=1 -> x likely chaotic
    """
    N = len(x)
    j = np.arange(1, N+1)
    t = np.arange(1, round(N/10) + 1)
    M = np.zeros(round(N/10))
    c = np.pi / 5 + np.random.rand(100) * 3 * np.pi / 5;  # 100 random c values in [pi/5,4pi/5]
    logging.debug("x: {}, N: {}, j: {}, t: {}, M: {}, c: {}".format(x.shape, N, j.shape, t.shape, M.shape, c.shape))
    kcorr = np.zeros(100)
    for its in range(100):
        p = np.cumsum(x * np.cos(j * c[its]))
        q = np.cumsum(x * np.sin(j * c[its]))

        for n in range(round(N / 10)):
            tmp = np.mean((p[n + 1:N] - p[1:N - n]) ** 2 + (q[n + 1:N] - q[1:N - n]) ** 2) - np.mean(x) ** 2 * (
                        1 - np.cos(n * c[its])) / (1 - np.cos(c[its]))
            M[n] = tmp

        kcorr[its] = pearsonr(t, M)[0]

    a = (max(x) - min(x)) / np.mean(np.abs(np.diff(x))) > 10
    b = (np.median(kcorr[c < np.mean(c)]) - np.median(kcorr[c > np.mean(c)])) > 0.5
    if a or b:
        if show_warnings:
            logging.warning('Warning: data is probably oversampled.')
            logging.warning('Use coarser sampling or reduce the maximum value of c.')
            return None
        else:
            pass

    if plotting:
        plt.figure()
        plt.plot(c, kcorr, 'o')
        plt.xlabel('c'), plt.ylabel('k')
        plt.show()
        plt.figure()
        plt.plot(t, M)
        plt.xlabel('t'), plt.ylabel('M')
        plt.show()
    return np.median(kcorr)


