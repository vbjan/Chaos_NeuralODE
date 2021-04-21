'''

    Version: 13
    TODO:
        - catching data problems in trainer

    Learnings:
        - ANODE becomes unstable for chaotic systems. Penalty by putting augmented dimensions in loss function,
        creates periodic behavior.
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
from utils import get_lr
from utils import load_h5_data, z1test, split_sequence, make_batches_from_stack
from utils import save_models, load_models, save_optimizer, load_optimizer, get_num_trainable_params
from Datasets import DDDLorenzData, VanDePol


class Net(nn.Module):
    '''
    The NN that learns the right hand side of the ODE
    '''
    def __init__(self, hidden_dim, io_dim, time_dependent=False):
        super().__init__()       # Run init of parent class
        self.io_dim = io_dim
        self.time_dependent = time_dependent
        self.acti = nn.LeakyReLU()
        #self.acti = nn.Tanh()

        if self.time_dependent:
            self.layer1 = nn.Linear(self.io_dim + 1, hidden_dim)
        else:
            self.layer1 = nn.Linear(self.io_dim, hidden_dim)

        #self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, self.io_dim)

    def forward(self,t , x):
        if self.time_dependent:
            t_vec = torch.ones(x.shape[0], 1) * t
            t_and_x = torch.cat([t_vec, x], 1)
            x = self.acti(self.layer1(t_and_x))
        else:
            x = self.acti(self.layer1(x))

        #x = self.acti(self.layer2(x))
        x = self.layer3(x)
        return x


def get_approx_k_of_model(model, val_data):
    """
    :param model: NN that learnt the systems ODE
    :param val_data: validation data with dimensions (batches, trajectory length, dim of system)
    :return: approximate K of model according to 0-1 test
    """
    index = np.random.randint(0, len(val_data))
    ic = torch.from_numpy(val_data[index][0, :]).float().view(1, 3)
    dt_val = 0.01
    times = torch.arange(0, 1, dt_val)
    traj = odeint(model, ic, times).view(-1, 3).detach().numpy()
    model_ks = []
    for i in range(3):
        model_ks.append(z1test(traj[:, i], show_warnings=False))
    model_k = np.mean(model_ks)
    return model_k


# Lorenz Models:
class LorenzModel(nn.Module):
    """
        Model of the real Lorenz System
    """
    def __init__(self,sigma = 10, rho = 28, beta=8./3):
        super(LorenzModel, self).__init__()
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def forward(self, t, x):
        dxdt = torch.zeros(x.size())
        dxdt[:, 0] = self.sigma * (x[:, 1] - x[:, 0])
        dxdt[:, 1] = x[:, 0] * (self.rho - x[:, 2]) - x[:, 1]
        dxdt[:, 2] = x[:, 0] * x[:, 1] - self.beta * x[:, 2]
        return dxdt


class PeriodicApprox(nn.Module):
    """
        Model of Lorenz Approximation, that is periodic
    """
    def __init__(self):
        super(PeriodicApprox, self).__init__()
        self.temp = 0

    def forward(self, t, x):
        dxdt = torch.zeros(x.size())
        dxdt[:, 0] = -9.913 * x[:, 0] + 9.913 * x[:, 1]
        dxdt[:, 1] = -7.175 * x[:, 0] + 20.507 * x[:, 1] - 0.613 * x[:, 1] * x[:, 2]
        dxdt[:, 2] = -3.05 * x[:, 2] + 0.504 * x[:, 0] ** 2 + 0.479 * x[:, 1] ** 2
        return dxdt


class KnowledgeModel(nn.Module):
    def __init__(self, hidden_dim, known_model, io_dim):
        super().__init__()       # Run init of parent class
        self.acti = nn.LeakyReLU()
        self.io_dim = io_dim
        self.layer1 = nn.Linear(self.io_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, self.io_dim)
        self.layer3 = nn.Linear(2*self.io_dim, self.io_dim)
        self.known_model = known_model

    def forward(self, t, x):
        # prediction of NN
        x_net = self.acti(self.layer1(x))
        x_net = self.layer2(x_net)

        # prediction of known model
        temp = torch.zeros(1)
        x_known_model = self.known_model(temp, x)

        x_combined = torch.cat((x_net, x_known_model), dim=1)
        x = self.layer3(x_combined)
        return x


# Van de Pol Models
class VanDePolModel(nn.Module):
    def __init__(self, mu=0.2):
        super(VanDePolModel, self).__init__()
        self.mu = mu
        self.temp = nn.Linear(1, 1)

    def forward(self, t, x):
        dxdt = torch.zeros(x.size())
        dxdt[:, 0] = x[:, 1]
        dxdt[:, 1] = -x[:, 0] + self.mu * (x[:, 1] - x[:, 0] ** 2 * x[:, 1]) + 0.32 * torch.sin(1.15 * t)
        #dxdt[:, 2] = 1
        return dxdt


class Trainer():
    def __init__(self, model, optimizer, train_dataset, val_dataset, model_dir, epochs, batch_size, lr, shuffle=False,
                 train_on_one_batch=False):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.data_dim = self.train_dataset.data_dim
        self.model_dir = model_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.shuffle = shuffle
        self.train_on_one_batch = train_on_one_batch

    def train(self):
        lookahead = train_dataset.lookahead
        val_lookahead = val_dataset.lookahead
        t_val = torch.linspace(0, val_lookahead * dt, val_lookahead + 1)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=self.shuffle, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=self.shuffle, drop_last=True)

        val_losses = []
        train_losses = []
        pre_val_loss = 1000000

        n_iterations = int(len(self.train_dataset) / self.batch_size)
        logging.info("\nSTARTING TO TRAIN MODEL | EPOCHS : {} | lr: {} | batchsize: {} | lookahead: {}"
                     .format(self.epochs, self.lr, self.batch_size, lookahead))
        logging.info("                        | iterations: {} | #trainable parameters: {} \n"
                     .format(n_iterations, get_num_trainable_params(params)))

        pre_train_time = time.time()
        now = pre_train_time
        for epoch in range(self.epochs):

            for n, (batch_X0, batch_X) in enumerate(train_dataloader):
                self.model.train()
                self.optimizer.zero_grad()
                batch_X0 = batch_X0.view(self.batch_size, self.data_dim)
                X_pred = odeint(f, batch_X0, t).permute(1, 0, 2)
                loss = torch.mean(torch.abs(X_pred - batch_X) ** 2)
                loss.backward()
                self.optimizer.step()
                if self.train_on_one_batch:
                    break

            train_losses.append(float(loss.detach()))
            self.model.eval()
            val_X0, val_X = next(iter(val_dataloader))
            val_X0 = val_X0.view(self.batch_size, self.data_dim)
            X_val_pred = odeint(self.model, val_X0, t_val).permute(1, 0, 2)
            val_loss = torch.mean(torch.abs(X_val_pred - val_X) ** 2)
            val_losses.append(float(val_loss))

            # scheduler.step(val_loss)

            if epoch % 2 == 0:
                time_for_epochs = int(time.time() - now)
                logging.info("EPOCH {} finished with training loss: {} | validation loss: {} | lr: {} | delta time: {}s"
                             .format(epoch, loss, val_loss, get_lr(self.optimizer), time_for_epochs))
                save_models(self.model_dir, self.model)
                save_optimizer(self.model_dir, self.optimizer)
                # if val_loss > pre_val_loss and EPOCH % 30 == 0:
                #    logging.info("\n STOPPING TRAINING EARLY BECAUSE VAL.LOSS STOPPED IMPROVING!\n")
                #    break
                now = time.time()
            pre_val_loss = val_loss

        post_train_time = time.time()
        logging.info('\nTRAINING FINISHED after {} seconds'.format(int(post_train_time - pre_train_time)))
        plt.plot(np.log(train_losses), label='train loss')
        plt.plot(np.log(val_losses), label='validation loss')
        plt.legend(), plt.savefig(figures_dir + '/losses.png'), plt.show()


def test_lorenz():
    # example plot of data
    x = ic_future[:, :, 0].view(-1).detach().numpy()
    y = ic_future[:, :, 1].view(-1).detach().numpy()
    z = ic_future[:, :, 2].view(-1).detach().numpy()
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, '-', label='Real Lorenz')
    ax.plot3D(ex_traj[:, 0], ex_traj[:, 1], ex_traj[:, 2], '-', label="Learnt Lorenz")
    ax.legend()
    plt.savefig(figures_dir + "/lorenz3d.png")
    plt.show()

    data_time = np.linspace(0, N*dt, N+1)
    # compare x, y and z of real and learnt trajectory
    plt.figure()
    plt.plot(data_time, x, '-', label='Real x')
    plt.plot(t, ex_traj[:, 0], label='Learnt x')
    plt.xlabel('time t')
    plt.legend()
    plt.savefig(figures_dir + "/lorenzx.png")
    plt.show()

    plt.figure()
    plt.plot(data_time, y, '-', label='Real y')
    plt.plot(t, ex_traj[:, 1], label='Learnt y')
    plt.xlabel('time t')
    plt.legend()
    plt.savefig(figures_dir + "/lorenzy.png")
    plt.show()

    plt.figure()
    plt.plot(data_time, z, '-', label='Real z')
    plt.plot(t, ex_traj[:, 2], label='Learnt z')
    plt.xlabel('time t')
    plt.legend()
    plt.savefig(figures_dir + "/lorenzz.png")
    plt.show()

    # 0-1 Test for chaos in real and learnt trajectories
    subsample = 13
    logging.info("\nCalculating K on trajectories of length : {} and {}".format(len(x), len(ex_traj[:, 0])))
    logging.info("0-1 test of | real x: {} | learnt x: {}".format(z1test(x[::subsample]), z1test(ex_traj[::subsample, 0])))
    logging.info("            | real y: {} | learnt y: {}".format(z1test(y[::subsample]), z1test(ex_traj[::subsample, 1])))
    logging.info("            | real z: {} | learnt z: {}".format(z1test(z[::subsample]), z1test(ex_traj[::subsample, 2])))


if __name__ == "__main__":

    # logging settings
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    logging.basicConfig(filename="3D_lorenz_prediction/ANODE.log", level=logging.INFO,
                        format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info("\n ----------------------- Starting of new script ----------------------- \n")

    data_name = 'VanDePol'
    #data_name = 'Lorenz'

    project_dir = os.path.dirname(os.path.realpath(__file__))
    if data_name == 'VanDePol':
        dt = 0.02
        test_data_dir = project_dir + '/data/VanDePol/van_de_pol_test_data_nt.pt'
        train_data_dir = project_dir + '/data/VanDePol/van_de_pol_train_data_nt.pt'
        val_data_dir = project_dir + '/data/VanDePol/van_de_pol_val_data_nt.pt'
        model_dir = project_dir + '/van_de_pol_prediction/models/vdp'
        figures_dir = project_dir + "/van_de_pol_prediction/figures"
        dataset = VanDePol
        data_dim = 2

    elif data_name == 'Lorenz':
        # directory settings
        dt = 0.01    # read out from simulation script
        test_data_dir = project_dir + "/data/Data3D" + str(dt) + "/test/data.h5"
        train_data_dir = project_dir + "/data/Data3D" + str(dt) + "/train/data.h5"
        val_data_dir = project_dir + "/data/Data3D" + str(dt) + "/val/data.h5"
        model_dir = project_dir + '/3D_lorenz_prediction/models/knowledge'
        # model_dir = project_dir + '/3D_lorenz_prediction/models/3DLorenzmodel'
        figures_dir = project_dir + "/3D_lorenz_prediction/figures"
        dataset = DDDLorenzData
        data_dim = 3

    # data settings
    lookahead = 2
    batch_size = 500
    t = torch.linspace(0, lookahead*dt, lookahead+1)
    print(t)
    assert(len(t) == lookahead + 1)
    max_len = 600

    # model settings
    TRAIN_MODEL = False
    LOAD_THEN_TRAIN = False
    EPOCHS = 150
    LR = 0.01

    # construct approximation model
    if data_name == 'Lorenz':
        mistake_factor = 0.9
        sigma = mistake_factor * 10
        beta = 8./3
        rho = 28
        #approx_model = LorenzModel(sigma, rho, beta)
        #approx_model = PeriodicApprox()
    elif data_name == 'VanDePol':
        mistake_factor = 0.9
        mu = mistake_factor*0.2
        approx_model = VanDePolModel(mu=mu)
    approx_model.eval()

    # constructing NN model that learns the dynamics
    f = Net(hidden_dim=100, io_dim=data_dim, time_dependent=False)
    #f = VanDePolModel()
    #f = KnowledgeModel(hidden_dim=100, known_model=approx_model, io_dim=data_dim)
    logging.info(f)

    params = (list(f.parameters()))
    optimizer = optim.Adam(params, lr=LR)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=False, min_lr=LR/100)

    if TRAIN_MODEL:
        if LOAD_THEN_TRAIN:
            load_models(model_dir, f)
            #load_optimizer(model_dir, optimizer)

        train_dataset = dataset(train_data_dir, lookahead=lookahead, tau=1, k=1, max_len=max_len)

        val_lookahead = 20
        val_dataset = dataset(val_data_dir, lookahead=val_lookahead, tau=1, k=1, max_len=max_len)

        trainer = Trainer(
            model=f,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_dir=model_dir,
            epochs=EPOCHS,
            lr=LR,
            batch_size=batch_size,
            shuffle=True,
            train_on_one_batch=False
        )
        trainer.train()

    else:
        load_models(model_dir, f)
        load_optimizer(model_dir, optimizer)

    with torch.no_grad():

        N = 1000
        test_dataset = dataset(test_data_dir, lookahead=N, tau=1, k=1, max_len=int(1.5*N))
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
        ic_state, ic_future = next(iter(test_dataloader))
        ic_state = ic_state.view(1, data_dim)
        dt_test = dt
        N_pred = int(N*dt/dt_test)
        t = torch.arange(0, N_pred*dt_test, dt_test)
        logging.info('Calculating test prediction...')
        g = VanDePolModel(mu=10*0.2)
        ex_traj = np.array(odeint(f, ic_state, t).view(-1, data_dim))
        logging.info('Finished test prediction...')

        if data_name == 'Lorenz':
            test_lorenz()

        if data_name == 'VanDePol':
            x = ic_future[:, :, 0].view(-1).detach().numpy()
            y = ic_future[:, :, 1].view(-1).detach().numpy()

            plt.plot(x, y, label='Real')
            plt.plot(ex_traj[:, 0], ex_traj[:, 1], label='Learnt')
            plt.title('Van de Pol oscillator')
            plt.legend()
            plt.show()




    logging.info("\n REACHED END OF MAIN")

