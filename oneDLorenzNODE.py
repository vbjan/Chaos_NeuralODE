import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import os
import logging
from utils import get_lr
from utils import save_models, load_models, load_h5_data, z1test, CreationRNN
from threeDLorenzNODE import recast_sequence_to_batches

os.environ['KMP_DUPLICATE_LIB_OK']='True'
logging.basicConfig(filename="1D_lorenz_prediction/1DLorenzNODE.log", level=logging.INFO,
                    format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
#logging.getLogger('').addHandler(console)


class Net(nn.Module):
    '''
    The NN that learns the right hand side of the ODE
    '''
    def __init__(self, hidden_dim=100):
        super().__init__()       # Run init of parent class
        self.io_dim = 3
        #self.acti = nn.Tanh()
        self.acti = nn.LeakyReLU()
        self.layer1 = nn.Linear(self.io_dim, hidden_dim)
        #self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, self.io_dim)

    def forward(self,t , x):
        x = self.acti(self.layer1(x))
        #x = self.acti(self.layer2(x))
        x = self.layer3(x)
        return x


# TODO: make function more general...
def recast_sequence_to_batches2(sequence, lookpa, lookah):
    """
        :param sequence: numpy array with trajectory
        :param lookpa: timesteps to look into the past
        :param lookah: lookahead for input output split
        :return: np array with all the states and np stack with the resulting trajectories
    """
    N = len(sequence)

    x0s = sequence[0::lookah + 1]
    xis = []
    xs = []
    for i in range(N):
        if i % (lookah + 1) == 0:
            xis.append(sequence[i + 1:i + lookah + 1:1])
            xs.append(sequence[i:i + lookah + 1:1])
    # Cut last x0 if sequence was not long enough
    xis = np.stack(xis[:-1])
    xs = np.stack(xs[:-1])
    if x0s.shape[0] != xis.shape[0]:
        x0s = x0s[:-1]
    # logging.info(len(x0s), len(xis), len(xs))
    assert len(x0s) == len(xis) and len(x0s) == len(xs)
    return x0s, xis, xs  # x0s.dim=(n_batches,3), xis.dim=(n_batches, lookah, 3)


if __name__ == "__main__":
    logging.info("\n ----------------------- Starting of new script ----------------------- \n")
    project_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = project_dir + "/1D_lorenz_prediction/Data/test/data.h5"
    train_data_dir = project_dir + "/1D_lorenz_prediction/Data/train/data.h5"
    val_data_dir = project_dir + "/1D_lorenz_prediction/Data/val/data.h5"
    figures_dir = project_dir + "/1D_lorenz_prediction/figures"
    model_dir = project_dir + '/1D_lorenz_prediction/models/3DLorenzmodel'

    # Load in the data
    d_train = load_h5_data(train_data_dir)
    logging.info("Shape of d_train: {}".format(d_train.shape))
    d_val = load_h5_data(val_data_dir)
    d_test = load_h5_data(test_data_dir)
    n_of_data = len(d_train[1])
    dt = 0.0025  # read out from simulation script
    lookahead = 2
    n_of_blocks = 8
    t = torch.from_numpy(np.arange(0, (1 + lookahead) * dt, dt))
    # TODO: get nbatches from data

    # Settings
    TRAIN_MODEL = True
    LOAD_THEN_TRAIN = False
    EPOCHS = 3000
    LR = 0.01
    HIDDEN_DIM = 6

    # Construct model
    encoder_rnn = CreationRNN(
                            input_dim=1,
                            hidden_dim=HIDDEN_DIM,
                            num_layers=1,
                            output_dim=3,
                            nbatch=None
                            )
    f = Net(hidden_dim=256)
    logging.info(encoder_rnn)
    logging.info(f)

    params = list(f.parameters())
    optimizer = optim.Adam(params, lr=LR)

    if TRAIN_MODEL:
        logging.info(d_train[1, :, :].shape)
        X0, Xi, X = recast_sequence_to_batches(d_train[1, :, :], lookahead)
        print(X0.shape, Xi.shape, X.shape)

