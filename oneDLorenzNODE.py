'''
    Version: 2.0
    By: Jan-Philipp von Bassewitz
    Bachelor Thesis: Learning Effective Neural ODE Dynamics

    Goal: By observing only one components history of the Lorenz attractor,
            predict the future of that component (see: Takens Theorem)
    Approach: Jointly train a GRU that encodes the history of the trajectory into a predict current 3D initial condition
            and a Neural ODE that predicts the future evolution of this 3D initial condition. Then map the prediction
            back into 1D observation space.

    Notes:
        - Do not shuffle data! It messes up the construction of the U_hat tensor, which relies on chronological order

    TODO:
        - Minimize wrt to weights and also the trajectory... see utils
        - Map predicted trajectory to x -> NODE space becomes independent of real space and might allow for better model

'''

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import time
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint

from utils import get_lr, get_num_trainable_params
from utils import save_models, load_models, save_optimizer, load_optimizer, CreationRNN, z1test
from Datasets import DDDLorenzData


class Net(nn.Module):
    '''
    The NN that learns the right hand side of the ODE
    '''
    def __init__(self, latent_dim=3, hidden_dim=100):
        super().__init__()       # Run init of parent class
        self.io_dim = latent_dim
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


class Decoder(nn.Module):
    def __init__(self, latent_dim=3, hidden_dim=100):
        super().__init__()       # Run init of parent class
        self.i_dim = latent_dim
        #self.acti = nn.Tanh()
        self.acti = nn.ReLU()
        self.layer1 = nn.Linear(self.i_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.acti(self.layer1(x))
        #x = self.acti(self.layer2(x))
        x = self.layer3(x)
        return x


def predict_next_steps(hist):
    # encode ic state
    hid = torch.zeros(rnn_num_layers, 1, rnn_hidden_dim)
    for j in range(k):
        x = hist[:, j, :].view(1, 1, 1)
        out, hid = encoder_rnn.forward(x, hid)
    U0 = out.view(-1, latent_dim)

    dt_test = dt
    t = torch.arange(0, 3 * dt_test, dt_test)
    ex_traj = odeint(f, U0, t).view(-1, latent_dim)
    return ex_traj[:, 0].view(1, 3, 1)


def predict_trajectory(hist, t):
    # encode history into 3D initial condition vector (out)
    batches = hist.size(0)
    print("bathces: ", batches, hist.size())
    hid = torch.zeros(encoder_rnn.num_layers, batches, encoder_rnn.hidden_dim)

    for j in range(k):
        x = hist[:, j, :].view(batches, 1, 1)
        out, hid = encoder_rnn.forward(x, hid)

    # print("out: ", out.size())
    U0 = out.view(-1, latent_dim)
    U_pred = odeint(f, U0, t).permute(1, 0, 2)
    # print('U_pred: ', U_pred.size())

    U_t_hat = U_pred[:-lookahead, :, :]
    # print('U_t_hat: ', U_t_hat.size())

    # restricting the latent space mapping to stay in the same space
    U_t = torch.zeros(batches - lookahead, lookahead + 1, latent_dim)
    # print("U_t: ", U_t.size())
    for j in range(batches - lookahead):  # iterating through the batches
        for n in range(lookahead + 1):  # iterating through the trajectory
            U_t[j, n, :] = out[j + n, :, :].view(-1)

    x_t_hat = U_t_hat[:, :, 0].view(-1, lookahead + 1, 1)

    return U_t_hat, x_t_hat, U_t



if __name__ == "__main__":

    # logging settings
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    logging.basicConfig(filename="1D_lorenz_prediction/1DLorenzNODE.log", level=logging.INFO,
                        format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info("\n ----------------------- Starting of new script ----------------------- \n")

    # directory settings
    project_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = project_dir + "/data/Data1D0.005/test/data.h5"
    train_data_dir = project_dir + "/data/Data1D0.005/train/data.h5"
    val_data_dir = project_dir + "/data/Data1D0.005/val/data.h5"
    figures_dir = project_dir + "/1D_lorenz_prediction/figures"
    model_dir = project_dir + '/1D_lorenz_prediction/models/3DLorenzmodel'

    # data settings
    dt = 0.005   # read out from simulation script
    lookahead = 2
    k = 7
    tau = 1
    augmented_dim = 2
    latent_dim = 1 + augmented_dim
    force = 0.1
    n_groups = 1
    batch_size = 300
    t = torch.from_numpy(np.arange(0, (1 + lookahead) * dt, dt))

    # Settings
    TRAIN_MODEL = False
    LOAD_THEN_TRAIN = False
    EPOCHS = 200
    LR = 0.001
    rnn_hidden_dim = 200
    rnn_num_layers = 1

    # Construct model
    encoder_rnn = CreationRNN(
                            input_dim=1,
                            hidden_dim=rnn_hidden_dim,
                            num_layers=rnn_num_layers,
                            output_dim=latent_dim,
                            nbatch=batch_size
                            )
    f = Net(latent_dim=latent_dim, hidden_dim=256)
    logging.info(encoder_rnn)
    logging.info(f)


    params = list(f.parameters()) + list(encoder_rnn.parameters())
    optimizer = optim.Adam(params, lr=LR)

    if TRAIN_MODEL:
        if LOAD_THEN_TRAIN:
            load_optimizer(model_dir, optimizer)
            load_models(model_dir, f, encoder_rnn)

        train_dataset = DDDLorenzData(train_data_dir, lookahead=lookahead, tau=tau, k=k, n_groups=n_groups, dim=1)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        val_dataset = DDDLorenzData(val_data_dir, lookahead=lookahead, tau=tau, k=k, n_groups=n_groups, dim=1)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        val_losses = []
        train_losses = []
        now = time.time()
        n_iterations = int(len(train_dataset)/batch_size)
        logging.info("\nSTARTING TO TRAIN MODEL")
        logging.info("EPOCHS : {} | lr: {} | batchsize: {} | lookahead: {} | iterations: {} | #trainable parameters: {}"
                     .format(EPOCHS, get_lr(optimizer), batch_size, lookahead, n_iterations, get_num_trainable_params(params)))
        for EPOCH in range(EPOCHS):

            for i, (history_Xs, future_Xs) in enumerate(train_dataloader):
                optimizer.zero_grad()

                # encode history into 3D initial condition vector (out)
                hid = encoder_rnn.init_hidden()
                for j in range(k):
                    x = history_Xs[:, j, :].view(batch_size, 1, 1)
                    out, hid = encoder_rnn.forward(x, hid)

                #print("out: ", out.size())
                U0 = out.view(-1, latent_dim)

                U_pred = odeint(f, U0, t).permute(1, 0, 2)
                #print('U_pred: ', U_pred.size())

                U_t_hat = U_pred[:-lookahead, :, :]
                #print('U_t_hat: ', U_t_hat.size())

                # TODO: out is not ordered in time!!!
                # restricting the latent space mapping to stay in the same space
                U_t = torch.zeros(batch_size - lookahead, lookahead+1, latent_dim)
                #print("U_t: ", U_t.size())
                for j in range(batch_size-lookahead):  # iterating through the batches
                    for n in range(lookahead+1):  # iterating through the trajectory
                        U_t[j, n, :] = out[j+n, :, :].view(-1)

                x_t = future_Xs[:-lookahead, :, :]
                x_t_hat = U_t_hat[:, :, 0].view(-1, lookahead+1, 1)

                #print("U_t: ", U_t.size())
                #print("x_t: {}, x_t_hat: {}".format(x_t.size(), x_t_hat.size()))

                loss = torch.mean(torch.abs(x_t - x_t_hat)**2) \
                       + force * torch.mean(torch.abs(U_t - U_t_hat)**2)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                #break

            if EPOCH % 2 == 0:
                time_for_epochs = time.time() - now

                # calculate validation loss
                #val_history_Xs, val_future_Xs = next(iter(val_dataloader))
                #U_val_t_hat, x_val_t_hat, U_val_t = predict_next_steps(val_history_Xs)
                #x_val_t = val_future_Xs[:-lookahead, :, :]
                #val_loss = torch.mean(torch.abs(x_val_t - x_val_t_hat)**2) + \
                #           force * torch.mean(torch.abs(U_val_t-U_val_t_hat)**2)
                #val_losses.append(float(val_loss))

                logging.info("EPOCH {} finished with training loss: {} | lr: {} | delta time: {}s"
                      .format(EPOCH, loss, get_lr(optimizer), time_for_epochs))
                save_models(model_dir, f, encoder_rnn)
                save_optimizer(model_dir, optimizer)
                now = time.time()

        print("TRAINING IS FINISHED")
        save_models(model_dir, f, encoder_rnn)
        save_optimizer(model_dir, optimizer)
        plt.plot(np.log(train_losses)), plt.show()
        #print("out.size: {}, U_pred.size: {}, U_t_hat.size: {}, U_t.size: {}".format(out.size(), U_pred.size(), U_t_hat.size(), U_t.size()))
        #print("x_t: {}, U_t[0]: {}, U_t: {}, U_t_hat: {}".format(x_t.size(), U_pred[:-2, 0, :].view(-1, lookahead+1, 1).size(), U_t.size(), U_t_hat.size() ))
    else:
        load_models(model_dir, f, encoder_rnn)
        load_optimizer(model_dir, optimizer)

    with torch.no_grad():

        # create and plot the testing trajectory
        N = 100
        test_dataset = DDDLorenzData(test_data_dir, lookahead=N, tau=tau, k=k, dim=1)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
        test_history_Xs, testX_future = next(iter(test_dataloader))

        # encode ic state
        hid = torch.zeros(rnn_num_layers, 1, rnn_hidden_dim)
        for j in range(k):
            x = test_history_Xs[:, j, :].view(1, 1, 1)
            out, hid = encoder_rnn.forward(x, hid)
        U0 = out.view(-1, latent_dim)

        # predict trajectory
        dt_test = dt
        t = torch.arange(0, (N+1)*dt_test, dt_test)
        N = len(t)
        U_ex_traj = odeint(f, U0, t).permute(1, 0, 2)
        x_traj = U_ex_traj[:, :, 0].view(-1)
        print(x_traj.size())
        ex_traj = np.array(U_ex_traj.view(-1, latent_dim))
        #x_traj = ex_traj[:, 0]

        ax = plt.axes(projection='3d')
        ax.plot3D(ex_traj[:, 0], ex_traj[:, 1], ex_traj[:, 2], '-', label="Learnt Lorenz")
        ax.legend()
        plt.show()

        # Compare x, y and z of real and learnt trajectory
        plt.figure()
        plt.plot(x_traj, label='Learnt x')
        plt.plot(testX_future.view(-1), '-o', label='Real x')
        plt.title("Observed Component")
        plt.xlabel("# steps")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(ex_traj[:, 1], label='Learnt y')
        plt.title("Unobserved Component")
        plt.xlabel("# steps")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(ex_traj[:, 2], label='Learnt z')
        plt.title("Unobserved Component")
        plt.xlabel("# steps")
        plt.legend()
        plt.show()

