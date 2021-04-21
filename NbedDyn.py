'''
    TODO:
        - Validation loss
        - implement Testing procedure
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
        #self.tanh = nn.Tanh()
        self.acti = nn.LeakyReLU()
        #self.acti = nn.ReLU()
        self.layer1 = nn.Linear(self.io_dim, hidden_dim)
        #self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        #self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, self.io_dim)

    def forward(self,t , x):
        x = self.acti(self.layer1(x))
        #x = self.acti(self.layer2(x))
        #x = self.acti(self.layer3(x))
        x = self.layer4(x)
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


if __name__ == "__main__":
    # logging settings
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    logging.basicConfig(filename="NbedDyn/NbedDyn.log", level=logging.INFO,
                        format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info("\n ----------------------- Starting of new script ----------------------- \n")

    # directory settings
    project_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = project_dir + "/data/old/Data1D0.005/test/data.h5"
    train_data_dir = project_dir + "/data/old/Data1D0.005/train/data.h5"
    val_data_dir = project_dir + "/data/old/Data1D0.005/val/data.h5"
    figures_dir = project_dir + "/NbedDyn/figures"
    model_dir = project_dir + '/NbedDyn/models/NbedDyn'

    # data settings
    dt = 0.005  # read out from simulation script
    lookahead = 1
    k = 7
    tau = 1
    augmented_dim = 3
    latent_dim = 1 + augmented_dim
    force = 3
    max_len = 1000     # to get longest possible data sequence
    batch_size = 200
    t = torch.from_numpy(np.arange(0, (1 + lookahead) * dt, dt))
    long_t = torch.arange(0, 25*dt, dt)

    # Settings
    TRAIN_MODEL = False
    LOAD_THEN_TRAIN = False
    EPOCHS = 600
    LR = 0.001

    train_dataset = DDDLorenzData(train_data_dir, lookahead=lookahead, tau=tau, k=k, max_len=max_len, dim=1, normalize=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    n_iterations = int(len(train_dataset) / batch_size)
    train_y_batches = []
    for n in range(n_iterations):
        train_y_batches.append(torch.rand((batch_size, 1, augmented_dim), requires_grad=True))

    f = Net(latent_dim=latent_dim, hidden_dim=1000)
    logging.info(f)
    params = list(f.parameters()) + list(train_y_batches)
    optimizer = optim.Adam(params, lr=LR)

    if TRAIN_MODEL:
        if LOAD_THEN_TRAIN:
            load_models(model_dir, f)
            load_optimizer(model_dir, optimizer)
            train_y_batches = torch.load(model_dir + 'train_y_batches.pt')

        train_losses = []

        logging.info("\nSTARTING TO TRAIN MODEL")
        logging.info("EPOCHS : {} | lr: {} | batchsize: {} | lookahead: {} | iterations: {} | #trainable parameters: {}"
                     .format(EPOCHS, get_lr(optimizer), batch_size, lookahead, n_iterations,
                             get_num_trainable_params(params)))

        pre_train_time = time.time()
        now = pre_train_time
        for EPOCH in range(EPOCHS):

            for i, (_, future_Xs) in enumerate(train_dataloader):
                optimizer.zero_grad()
                train_y_batch = train_y_batches[i]
                X0 = future_Xs[:, 0, 0].view(batch_size, 1, 1)
                U0 = torch.cat((X0, 2*train_y_batch-1), dim=2)  # 0th component is the x component!

                # predict one step ahead for each U0
                U_pred = odeint(f, U0.view(batch_size, latent_dim), t).permute(1, 0, 2)

                # build future Us out of the known U0s
                future_Us = torch.zeros((batch_size - lookahead, lookahead + 1, latent_dim))
                for j in range(batch_size - lookahead):  # iterating through the batches
                    for n in range(lookahead + 1):  # iterating through the trajectory
                        future_Us[j, n, :] = U0[j + n, :, :].view(-1)

                assert(future_Us.requires_grad == True)

                # bring everything to the same dimensions
                U_pred = U_pred[:-lookahead, :, :]
                X_pred = U_pred[:, :, 0].view(batch_size-lookahead, 1+lookahead, 1)
                future_Xs = future_Xs[:-lookahead, :, :].view(batch_size-lookahead, 1+lookahead, 1)

                '''print(future_Us[0, :, 0])
                print(U_pred.requires_grad)
                print(future_Us[:, :, 0].requires_grad)
                print(X_pred.requires_grad)
                print(future_Xs.requires_grad,'\n')'''

                a = torch.mean(torch.abs(X_pred - future_Xs)**2)
                b = torch.mean(torch.abs(U_pred - future_Us)**2)
                loss = a + force * b

                loss.backward()
                optimizer.step()

            train_losses.append(float(loss))
            if EPOCH % 10 == 0:
                time_for_epochs = time.time() - now
                logging.info("EPOCH {} finished with training loss: {} | lr: {} | delta time: {} s"
                            .format(EPOCH, loss, get_lr(optimizer), time_for_epochs))
                print("losses: a: {}, b: {}, total: {}".format(a, b, loss))

                save_models(model_dir, f)
                save_optimizer(model_dir, optimizer)
                torch.save(train_y_batches, model_dir + 'train_y_batches.pt')
                now = time.time()

        plt.plot(np.log(train_losses))
        print('Exmple U0:', U0.size())

        x_pred = U_pred[:, 1, 0].view(-1).detach().numpy()
        x = U0[1:, 0, 0].view(-1).detach().numpy()
        y = U0[1:, 0, 1].view(-1).detach().numpy()
        z = U0[1:, 0, 2].view(-1).detach().numpy()
        plt.show()
        plt.plot(x)
        plt.plot(x_pred)
        plt.show()
        plt.plot(y)
        plt.show()
        plt.plot(z)
        plt.show()
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z, '-', label='Real Lorenz')
        plt.show()
    else:
        #load_optimizer(model_dir, optimizer)
        load_models(model_dir, f)
        train_y_batches = torch.load(model_dir + 'train_y_batches.pt')
        print("LOADED MODEL")

    f.eval()
    # Testing the model:
    # create and plot the testing trajectory
    N = 200
    TEST_EPOCHS = 100
    TEST_LR = 0.02
    dt_test = dt   #TODO: saller dt for testing
    test_dataset = DDDLorenzData(test_data_dir, lookahead=lookahead, tau=tau, k=k, max_len=1000, dim=1, normalize=True)
    test_dataloader = DataLoader(test_dataset, batch_size=N, shuffle=False, drop_last=True)
    test_history_X, test_future_X = next(iter(test_dataloader))
    known_X = test_future_X[:k, :, :].view(k, lookahead+1, 1)

    # Find out U0 by minimizing loss w.r.t. unkown trajectory
    batch_y = [torch.rand((k, 1, augmented_dim), requires_grad=True)]
    test_params = list(batch_y)
    test_optimizer = optim.Adam(test_params, lr=TEST_LR)
    for EPOCH in range(TEST_EPOCHS):
        test_optimizer.zero_grad()

        X0 = known_X[:, 0, 0].view(k, 1, 1)
        U0 = torch.cat((X0, 2 * batch_y[0] - 1), dim=2)

        U_pred = odeint(f, U0.view(k, latent_dim), t).permute(1, 0, 2)

        future_Us = torch.zeros((k - lookahead, lookahead + 1, latent_dim))
        for j in range(k - lookahead):  # iterating through the batches
            for n in range(lookahead + 1):  # iterating through the trajectory
                future_Us[j, n, :] = U0[j + n, :, :].view(-1)

        U_pred = U_pred[:-lookahead, :, :]
        X_pred = U_pred[:, :, 0].view(k - lookahead, 1 + lookahead, 1)
        future_X = known_X[:-lookahead, :, :].view(k - lookahead, lookahead+1, 1)

        a = torch.mean(torch.abs(X_pred - future_X) ** 2)
        b = torch.mean(torch.abs(U_pred - future_Us) ** 2)
        test_loss = a + force * b

        test_loss.backward()
        test_optimizer.step()

        if EPOCH % 10:
            logging.info('Loss on test optimization: {} | TEST_EPOCH: {}'.format(test_loss, EPOCH))

    with torch.no_grad():
        f.eval()
        logging.info('The test optimization resulted in U0: {}'.format(U0[-1, 0, :]))
        U0 = U0[-1, 0, :]

        t_pred = torch.arange(0, N * dt_test, dt_test)
        U_pred = odeint(f, U0.view(1, latent_dim), t_pred).permute(1, 0, 2)

        X_pred = U_pred[0, :, 0]

        x = X_pred.view(-1).detach().numpy()
        y = U_pred[0, :, 1].view(-1).detach().numpy()
        z = U_pred[0, :, 2].view(-1).detach().numpy()
        w = U_pred[0, :, 3].view(-1).detach().numpy()

        plt.plot(x, label='predicted')
        plt.title('Predicted vs. True dynamics')
        plt.plot(test_future_X[k:, 0, :].view(-1), label='true')
        #plt.plot(test_history_X[:, 0, :])
        plt.legend()
        plt.show()

        plt.title('Hidden states of the Learnt system')
        plt.plot(y)
        plt.plot(z)
        plt.plot(w)
        plt.show()

        ax = plt.axes(projection='3d')
        plt.title('learnt 3D dynamics')
        ax.plot3D(x, y, z, '-', label='Real Lorenz')
        plt.show()



        '''for EPOCH in range(TEST_EPOCHS):
    
            test_optimizer.zero_grad()
    
            X0 = test_future_Xs[:, 0, 0].view(N, 1, 1)
            U0 = torch.cat((X0, test_y_batch), dim=2)
    
            # build future Us out of the known U0s
            test_future_Us = torch.zeros(N - lookahead, lookahead + 1, latent_dim)
            for j in range(N - lookahead):  # iterating through the batches
                for n in range(lookahead + 1):  # iterating through the trajectory
                    test_future_Us[j, n, :] = U0[j + n, :, :].view(-1)
    
            U_pred = odeint(f, U0.view(N, latent_dim), t).permute(1, 0, 2)
    
            U_pred = U_pred[:-1, :, :]
            X_pred = U_pred[:, :, 0].view(N - lookahead, 1 + lookahead, 1)
            test_future_Xs = test_future_Xs[:-1, :, :].view(N - lookahead, 1 + lookahead, 1)
    
            test_loss = torch.mean(torch.abs(X_pred - test_future_Xs) ** 2) + \
                        force * torch.mean(torch.abs(U_pred - test_future_Us) ** 2)
    
            test_loss.backward()
            test_optimizer.step()
            logging.info("TEST_EPOCH: {} | test_loss: {}".format(EPOCH, test_loss))'''
