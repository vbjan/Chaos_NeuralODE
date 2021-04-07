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
from utils import save_models, load_models, load_h5_data, z1test, CreationRNN, split_sequence

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


def extract_real_uts(u0s):
    uts = torch.zeros()


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
    k = 7
    tau = 1
    latent_dim = 3
    force = 0.1
    n_of_batches = 1
    max_blocks = 512
    t = torch.from_numpy(np.arange(0, (1 + lookahead) * dt, dt))

    # Settings
    TRAIN_MODEL = True
    LOAD_THEN_TRAIN = False
    EPOCHS = 1000
    LR = 0.001
    HIDDEN_DIM = 6

    # Construct model
    encoder_rnn = CreationRNN(
                            input_dim=1,
                            hidden_dim=HIDDEN_DIM,
                            num_layers=1,
                            output_dim=latent_dim,
                            nbatch=max_blocks
                            )
    f = Net(hidden_dim=256)
    logging.info(encoder_rnn)
    logging.info(f)

    params = list(f.parameters()) + list(encoder_rnn.parameters())
    optimizer = optim.Adam(params, lr=LR)

    if TRAIN_MODEL:
        logging.info(d_train[1, :, :].shape)
        history_Xs, future_Xs = split_sequence(
                                    d_train[1, :, :],
                                    lookahead,
                                    tau=tau,
                                    k=k,
                                    max_blocks=max_blocks
                                    )
        logging.info(history_Xs.size())
        logging.info(future_Xs.size())
        val_losses = []
        train_losses = []

        for EPOCH in range(EPOCHS):
            optimizer.zero_grad()

            # encode history into 3D initial condition vector (out)
            hid = encoder_rnn.init_hidden()
            for i in range(k):
                x = history_Xs[:, i, :].view(max_blocks, 1, 1)
                out, hid = encoder_rnn.forward(x, hid)

            U0 = out.view(-1, latent_dim)
            U_pred = odeint(f, U0, t).permute(1, 0, 2)
            U_t_hat = U_pred[:-2, 1:lookahead+1, :] # the next predicted latent steps

            # restricting the latent space mapping to stay in the same space
            U_t = torch.zeros(max_blocks - lookahead, lookahead, latent_dim)
            for i in range(max_blocks-lookahead):
                U_t[i, 0, :] = out[i, :, :].view(-1)
                U_t[i, 1, :] = out[i+1, :, :].view(-1)
            x_t = future_Xs[:-2, :, :]

            loss = torch.mean(torch.abs(x_t - U_pred[:-2, 0, :].view(-1, lookahead+1, 1))) \
                   + force * torch.mean(torch.abs(U_t - U_t_hat))
            train_losses.append(loss)
            loss.backward()
            optimizer.step()

            if EPOCH % 10 == 0:
                logging.info("EPOCH {} finished with training loss: {} | lr: {} \n"
                      .format(EPOCH, loss, get_lr(optimizer)))

        print("TRAINING IS FINISHED")
        save_models(model_dir, optimizer, f, encoder_rnn)
        plt.plot(train_losses), plt.show()
        #print("out.size: {}, U_pred.size: {}, U_t_hat.size: {}, U_t.size: {}".format(out.size(), U_pred.size(), U_t_hat.size(), U_t.size()))
        #print("x_t: {}, U_t[0]: {}, U_t: {}, U_t_hat: {}".format(x_t.size(), U_pred[:-2, 0, :].view(-1, lookahead+1, 1).size(), U_t.size(), U_t_hat.size() ))
    else:
        load_models(model_dir, optimizer, f, encoder_rnn)

    with torch.no_grad():
        ic = torch.tensor([10, 1, 1]).float()
        dt_test = 0.0025
        t = torch.arange(0, 0.5, dt_test)
        N = len(t)
        ex_traj = np.array(odeint(f, ic, t).view(-1, latent_dim))
        x = d_test[0, :, 0]

        ax = plt.axes(projection='3d')
        ax.plot3D(ex_traj[:, 0], ex_traj[:, 1], ex_traj[:, 2], '-', label="Learnt Lorenz")
        ax.legend()
        plt.show()

        # Compare x, y and z of real and learnt trajectory
        plt.figure()
        plt.plot(ex_traj[:, 0], label='Learnt x')
        plt.plot(x[:N], label='Real x')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(ex_traj[:, 1], label='Learnt y')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(ex_traj[:, 2], label='Learnt z')
        plt.legend()
        plt.show()

