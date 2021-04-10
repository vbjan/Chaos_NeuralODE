'''
Version 2.0 by Jan-Philipp von Bassewitz
TODO:
    -
'''

import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import os
import logging
from utils import get_lr, load_models, save_models, save_optimizer, load_optimizer
from utils import load_h5_data, z1test, CreationRNN, split_sequence
from Datasets import DDDLorenzData


def predict_state(history_x):
    # encode history into 3D initial condition vector (out)
    hid = encoder_rnn.init_hidden()
    for i in range(k):
        x = history_x[:, i, :].view(-1, 1, 1)
        out, hid = encoder_rnn.forward(x, hid)
    return out


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
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, self.io_dim)

    def forward(self,t , x):
        x = self.acti(self.layer1(x))
        x = self.acti(self.layer2(x))
        x = self.layer3(x)
        return x


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(filename="IdentifyIC/IdentifyIC.log", level=logging.INFO,
                        format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    project_dir = os.path.dirname(os.path.realpath(__file__))
    dddtest_data_dir = project_dir + "/IdentifyIC/Data3D0.01/test/data.h5"
    dddtrain_data_dir = project_dir + "/IdentifyIC/Data3D0.01/train/data.h5"
    dddval_data_dir = project_dir + "/IdentifyIC/Data3D0.01/val/data.h5"
    figures_dir = project_dir + "/IdentifyIC/figures"
    model_dir = project_dir + '/IdentifyIC/models/3DLorenzmodel'
    ddd_model_dir = project_dir + '/IdentifyIC/models/3Dmodel/3DLorenzmodel'

    dt = 0.01  # read out from simulation script
    k = 8
    tau = 1
    latent_dim = 3
    lookahead = 0
    batch_size = 200
    t = torch.from_numpy(np.arange(0, (1 + lookahead) * dt, dt))

    # Settings
    TRAIN_MODEL = False
    LOAD_THEN_TRAIN = False
    EPOCHS = 35
    LR = 0.01
    HIDDEN_DIM = 500
    num_rnn_layers = 2

    # Construct model
    encoder_rnn = CreationRNN(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        num_layers=num_rnn_layers,
        output_dim=latent_dim,
        nbatch=batch_size
    )
    print(encoder_rnn)
    params = list(encoder_rnn.parameters())
    optimizer = optim.Adam(params, lr=LR)

    f = Net(hidden_dim=256)
    load_models(ddd_model_dir, f)

    if TRAIN_MODEL:

        train_dataset = DDDLorenzData(dddtrain_data_dir, lookahead=lookahead, tau=tau, k=8)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        val_dataset = DDDLorenzData(dddval_data_dir, lookahead, tau, k)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        train_losses = []
        val_losses = []

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=False,
                                                               min_lr=LR/100)

        if LOAD_THEN_TRAIN:
            load_optimizer(model_dir, optimizer)
            load_models(model_dir, encoder_rnn)

        for EPOCH in range(EPOCHS):

            for i, (history_Us, future_Us) in enumerate(train_dataloader):

                optimizer.zero_grad()

                # encode history into 3D initial condition vector (out)
                history_Xs = history_Us[:, :, 0].view(-1, k, 1)
                inferred_U = predict_state(history_Xs)

                loss = torch.mean(torch.abs(inferred_U - future_Us)**2)

                loss.backward()
                optimizer.step()

            train_losses.append(float(loss))
            val_history_Us, val_future_Us = next(iter(val_dataloader))
            val_history_Xs = val_history_Us[:, :, 0].view(-1, k, 1)
            inferred_U_val = predict_state(val_history_Xs)
            val_loss = torch.mean(torch.abs(inferred_U_val.detach() - val_future_Us.detach())**2)
            val_losses.append(float(val_loss))
            scheduler.step(val_loss)

            if EPOCH % 2 == 0:
                print("EPOCH {} finished with training loss: {} | validation loss: {} | lr: {} "
                      .format(EPOCH, loss, val_loss, get_lr(optimizer)))
                save_models(model_dir, encoder_rnn)
                save_optimizer(model_dir, optimizer)

        print("TRAINING IS FINISHED!")
        plt.plot(np.log(train_losses), label='train loss'), plt.plot(np.log(val_losses), label="validation loss")
        plt.legend(), plt.show()
        save_models(model_dir, encoder_rnn)
        save_optimizer(model_dir, optimizer)
    else:
        load_models(model_dir, encoder_rnn)
        load_optimizer(model_dir, optimizer)

    with torch.no_grad():

        # create and plot the testing trajectory
        N = 50
        test_dataset = DDDLorenzData(dddtest_data_dir, lookahead=N, tau=tau, k=k)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
        testU_history, testU_future = next(iter(test_dataloader))

        testU_ic = testU_future[:, 0, :]
        print("Real initial condition: {}".format(testU_ic))
        logging.debug("testU_history: {}, testu_future: {}".format(testU_history.size(), testU_future.size()))
        testx_history = testU_history[:, :, 0].view(1, k, 1)
        testx_future = testU_future[:, :, 0].view(1, -1, 1)
        t_history = np.linspace(0, tau*dt*k, k)
        t_future = np.linspace(tau*dt*k, dt*(k+N+1), N+1)
        plt.figure()
        plt.plot(t_history, testx_history.view(-1), 'ro-', label="known history of x")
        plt.plot(t_future, testx_future.view(-1), 'go-', label="unknown future of x")

        # encode history into 3D initial condition vector (out)
        hid = torch.zeros(num_rnn_layers, 1, HIDDEN_DIM)
        for i in range(k):
            x = testx_history[:, i, :].view(1, 1, 1)
            out, hid = encoder_rnn.forward(x, hid)
        print("Predicted initial condition: {}".format(out))
        print("Loss on example prediction: {}".format(torch.mean(torch.abs(out-testU_ic)**2)))

        # predict the behavior of the system with the initial condition
        t = torch.linspace(0, dt*(N+1), N+1)
        ex_traj = np.array(odeint(f, out, t).view(-1, 3))
        logging.debug("t: {}".format(t.size()))
        logging.debug("extraj {}".format(ex_traj.shape))
        logging.debug("t_future: {}".format(t_future.shape))

        plt.plot(t_future, ex_traj[:, 0], 'k', label='Prediction of x')
        plt.plot(tau*k*dt, out[:, :, 0].detach().float(), 'cx', label='inferred point')
        plt.title("Partially Observed Lorenz Prediction")
        plt.xlabel('time t')
        plt.ylabel('x-component of Lorenz system')
        plt.legend(), plt.show()

        print("0-1 test of | real x: {} | learnt x: {}".format(z1test(testx_future.view(-1).numpy()),
                                                                      z1test(ex_traj[:, 0])))