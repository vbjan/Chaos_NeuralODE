"""
    Created by: Jan-Philipp von Bassewitz, CSE-lab, ETH Zurich
    Learning effective NODE dynamics with and without prior knowledge
    Version: 13

    TODO:
        - Clearer settings area

    Learnings:
        - ANODE becomes unstable for chaotic systems. Penalty by putting augmented dimensions in loss function,
        creates periodic behavior.
"""

import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import argrelextrema
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
from Datasets import DDDLorenzData, VanDerPol


class Net(nn.Module):
    '''
    The NN that learns the right hand side of the ODE
    '''
    def __init__(self, hidden_dim, io_dim, time_dependent=False):
        super().__init__()       # Run init of parent class
        self.io_dim = io_dim
        self.time_dependent = time_dependent
        self.acti = nn.LeakyReLU()

        if self.time_dependent:
            self.layer1 = nn.Linear(self.io_dim + 1, hidden_dim)
        else:
            self.layer1 = nn.Linear(self.io_dim, hidden_dim)

        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, self.io_dim)

    def forward(self,t , x):
        if self.time_dependent:
            t_vec = torch.ones(x.shape[0], 1) * t
            t_and_x = torch.cat([t_vec, x], 1)
            x = self.acti(self.layer1(t_and_x))
        else:
            x = self.acti(self.layer1(x))

        x = self.acti(self.layer2(x))
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


# Prior Lorenz Models:
class LorenzModel(nn.Module):
    """
        Model of the real Lorenz System
    """
    def __init__(self, sigma=10, rho=28, beta=8./3):
        super(LorenzModel, self).__init__()
        self.temp = nn.Linear(1, 1)
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
        #self.temp = nn.Linear(1, 1)

    def forward(self, t, x):
        dxdt = torch.zeros(x.size())
        dxdt[:, 0] = -9.913 * x[:, 0] + 9.913 * x[:, 1]
        dxdt[:, 1] = -7.175 * x[:, 0] + 20.507 * x[:, 1] - 0.613 * x[:, 1] * x[:, 2]
        dxdt[:, 2] = -3.05 * x[:, 2] + 0.504 * x[:, 0] ** 2 + 0.479 * x[:, 1] ** 2
        return dxdt


# Van der Pol prior models
class VanDerPolModel(nn.Module):
    def __init__(self, mu=0.2, time_dependent=True):
        super(VanDerPolModel, self).__init__()
        self.mu = mu
        self.temp = nn.Linear(1, 1)
        self.time_dependent = time_dependent

    def forward(self, t, x):
        dxdt = torch.zeros(x.size())
        dxdt[:, 0] = x[:, 1]
        if self.time_dependent:
            dxdt[:, 1] = -x[:, 0] + self.mu * (x[:, 1] - x[:, 0] ** 2 * x[:, 1]) + 0.32 * torch.sin(1.15 * x[:, 2])
            dxdt[:, 2] = 1
        else:
            dxdt[:, 1] = -x[:, 0] + self.mu * (x[:, 1] - x[:, 0] ** 2 * x[:, 1])
            dxdt[:, 2] = 0
        return dxdt


# knowledge based ODEnets:
class KnowledgeModel(nn.Module):
    """
        Embedded knowledge based model
    """
    def __init__(self, hidden_dim, known_model, io_dim=3):
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
        x_known_model = self.known_model(t, x)

        x_combined = torch.cat((x_net, x_known_model), dim=1)
        x = self.layer3(x_combined)
        return x


class SumKnowledgeModel(nn.Module):
    """
        Added knowledge based model
    """
    def __init__(self, hidden_dim, known_model, io_dim=3):
        super().__init__()       # Run init of parent class
        self.acti = nn.LeakyReLU()
        self.io_dim = io_dim
        self.layer1 = nn.Linear(self.io_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, self.io_dim)
        self.known_model = known_model

    def forward(self, t, x):
        # prediction of NN
        x_net = self.acti(self.layer1(x))
        x_net = self.layer2(x_net)

        # prediction of known model
        x_known_model = self.known_model(t, x)

        x = x_known_model + x_net
        return x


class Trainer:
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

        assert(n_iterations == int(len(self.train_dataset) / self.batch_size))
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

            if epoch % 10 == 0:
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


# plot example testing predictions
def test_lorenz(show_prior=False):
    # example plot of data
    x = ic_future[:, :, 0].view(-1).detach().numpy()
    y = ic_future[:, :, 1].view(-1).detach().numpy()
    z = ic_future[:, :, 2].view(-1).detach().numpy()

    ex_x = ex_traj[:, 0]
    ex_y = ex_traj[:, 1]
    ex_z = ex_traj[:, 2]

    ap_x = approx_traj[:, 0]
    ap_y = approx_traj[:, 1]
    ap_z = approx_traj[:, 2]

    ncol = 1
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, '-', label='Ground truth 1')
    ax.plot3D(ex_x, ex_y, ex_z, '-', label="Ground truth 2")
    if show_prior:
        ax.plot3D(ap_x, ap_y, ap_z, '-', label="Prior model")
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=ncol)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    plt.savefig(figures_dir + "/fully_obs_3d.png", dpi=300,
                format='png',
                bbox_inches='tight')
    plt.show()

    data_time = np.linspace(0, N*dt, N+1)
    # compare x, y and z of real and learnt trajectory
    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(data_time, x, '-', label='Ground truth 1')
    plt.plot(shifted_t, ex_x, label='Ground truth 2')
    if show_prior:
        plt.plot(t, ap_x, label='Prior')
    plt.xlabel('time t')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=ncol)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.savefig(figures_dir + "/fully_obs_x.png",
                dpi=300,
                format='png',
                bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(data_time, y, '-', label='Ground truth')
    plt.plot(shifted_t, ex_y, label='Learnt')
    if show_prior:
        plt.plot(t, ap_y, label='Prior')
    plt.xlabel('time t')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=ncol)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$y$')
    plt.savefig(figures_dir + "/fully_obs_y.png", dpi=300,
                format='png',
                bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(data_time, z, '-', label='Ground truth')
    plt.plot(shifted_t, ex_z, label='Learnt')
    if show_prior:
        plt.plot(t, ap_z, label='Prior')
    plt.xlabel('time t')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=ncol)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$z$')
    plt.savefig(figures_dir + "/fully_obs_z.png", dpi=300,
                format='png',
                bbox_inches='tight')
    plt.show()

    # tent map
    plt.figure(figsize=(6, 6), dpi=300)
    temp_idx = argrelextrema(z, np.greater)
    z_max = z[temp_idx]
    zn = z_max[:-1]
    znplus1 = z_max[1:]
    plt.plot(zn, znplus1, 'o', label='Ground truth')
    #plt.plot(zn, zn, 'r-')

    temp_idx = argrelextrema(ex_z, np.greater)
    ex_z_max = ex_z[temp_idx]
    ex_zn = ex_z_max[:-1]
    ex_znplus1 = ex_z_max[1:]
    plt.plot(ex_zn, ex_znplus1, 'o', label='Learnt')

    if show_prior:
        temp_idx = argrelextrema(ap_z, np.greater)
        ap_z_max = ap_z[temp_idx]
        ap_zn = ap_z_max[:-1]
        ap_znplus1 = ap_z_max[1:]
        plt.plot(ap_zn, ap_znplus1, 'o', label='Prior')

    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=ncol)
    plt.xlabel(r'local_max($z_n$)')
    plt.ylabel(r'local_max($z_{n+1}$)')
    plt.savefig(figures_dir + "/fully_obs_cobweb.png", dpi=300,
                format='png',
                bbox_inches='tight')
    plt.show()

    # 0-1 Test for chaos in real and learnt trajectories
    subsample = 13
    logging.info("\nCalculating K on trajectories of length : {} and {}".format(len(x), len(ex_traj[:, 0])))
    logging.info("0-1 test of | real x: {} | learnt x: {}".format(z1test(x[::subsample]), z1test(ex_traj[::subsample, 0])))
    logging.info("            | real y: {} | learnt y: {}".format(z1test(y[::subsample]), z1test(ex_traj[::subsample, 1])))
    logging.info("            | real z: {} | learnt z: {}".format(z1test(z[::subsample]), z1test(ex_traj[::subsample, 2])))


def test_van_der_pol(show_prior=False):
    x = ic_future[:, :, 0].view(-1).detach().numpy()
    y = ic_future[:, :, 1].view(-1).detach().numpy()
    ncol = 1
    short_term = 100

    plt.figure(figsize=(6, 6))
    plt.plot(x[:short_term], y[:short_term], '-x', label='Ground truth')
    plt.plot(ex_traj[:short_term, 0], ex_traj[:short_term, 1], label='Learnt')
    if show_prior: plt.plot(approx_traj[:short_term, 0], approx_traj[:short_term, 1], label='Prior Approximation')
    #plt.title('Van der Pol oscillator: First {} steps'.format(str(short_term)))
    lg = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=ncol)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.savefig(figures_dir + "/" + str(mistake_factor) + "mu100steps.png",
                    dpi=300,
                    format='png',
                    bbox_extra_artists=(lg,),
                    bbox_inches='tight')
    plt.show()

    mid_term = 1000
    plt.figure(figsize=(6, 6))
    plt.plot(x[:mid_term], y[:mid_term], '-', label='Ground truth')
    plt.plot(ex_traj[:mid_term, 0], ex_traj[:mid_term, 1], label='Learnt')
    if show_prior: plt.plot(approx_traj[:mid_term, 0], approx_traj[:mid_term, 1], label='Prior Approximation')
    #plt.title('Van der Pol oscillator: First {} steps'.format(str(mid_term)))
    lg = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=ncol)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.savefig(figures_dir + "/" + str(mistake_factor) + "mu1000steps.png", dpi=300,
                    format='png',
                    bbox_extra_artists=(lg,),
                    bbox_inches='tight')
    plt.show()

    long_term = 10000
    plt.figure(figsize=(6, 6))
    plt.plot(x[:long_term], y[:long_term], '-', label='Ground truth')
    plt.plot(ex_traj[:long_term, 0], ex_traj[:long_term, 1], label='Learnt')
    if show_prior: plt.plot(approx_traj[:long_term, 0], approx_traj[:long_term, 1], label='Prior Approximation')
    #plt.title('Van der Pol oscillator: First {} steps'.format(str(long_term)))
    lg = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=ncol)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.savefig(figures_dir + "/" + str(mistake_factor) + "mu10ksteps.png", dpi=300,
                    format='png',
                    bbox_extra_artists=(lg,),
                    bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(x[:long_term], label='Ground truth')
    # plt.plot(ex_traj[:long_term, 0], label='Learnt')
    # if show_prior: plt.plot(approx_traj[:long_term, 0], label='Approx')
    # plt.legend()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.plot(y[:long_term], label='Ground truth')
    plt.plot(ex_traj[:long_term, 1], label='Learnt')
    if show_prior: plt.plot(approx_traj[:long_term, 1], label='Approx')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=ncol)
    plt.show()


if __name__ == "__main__":

    # logging settings
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    logging.basicConfig(filename="3D_lorenz_prediction/ANODE.log", level=logging.INFO,
                        format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info("\n ----------------------- Starting of new script ----------------------- \n")

    # set up matplot lib font size
    font = {'family': 'normal',
            'size': 17}
    matplotlib.rc('font', **font)

    # choose the system
    #data_name = 'VanDerPol'
    data_name = 'Lorenz'

    project_dir = os.path.dirname(os.path.realpath(__file__))
    if data_name == 'VanDerPol':
        dt = 0.02
        test_data_dir = project_dir + '/data/VanDerPol/van_de_pol_test_data.pt'
        train_data_dir = project_dir + '/data/VanDerPol/van_de_pol_train_data.pt'
        val_data_dir = project_dir + '/data/VanDerPol/van_de_pol_val_data.pt'
        model_dir = project_dir + '/van_de_pol_prediction/models/vdp'
        figures_dir = project_dir + "/van_de_pol_prediction/figures"
        dataset = VanDerPol
        data_dim = 3

    elif data_name == 'Lorenz':
        # directory settings
        dt = 0.01    # 0.01 or 0.1
        test_data_dir = project_dir + "/data/Data3D" + str(dt) + "/test/data.h5"
        train_data_dir = project_dir + "/data/Data3D" + str(dt) + "/train/data.h5"
        val_data_dir = project_dir + "/data/Data3D" + str(dt) + "/val/data.h5"
        #model_dir = project_dir + '/3D_lorenz_prediction/models/knowledge'
        model_dir = project_dir + '/3D_lorenz_prediction/models/3DLorenzmodel'
        figures_dir = project_dir + "/3D_lorenz_prediction/figures"
        dataset = DDDLorenzData
        data_dim = 3

    # data settings
    lookahead = 2
    batch_size = 500
    n_iterations = 1

    max_len = (n_iterations+1) * batch_size
    t = torch.linspace(0, lookahead*dt, lookahead+1)
    assert(len(t) == lookahead + 1)

    # model settings
    TRAIN_MODEL = False
    LOAD_THEN_TRAIN = False
    EPOCHS = 200
    LR = 0.01

    # construct approximation model
    if data_name == 'Lorenz':
        mistake_factor = 0.5
        sigma = mistake_factor * 10
        beta = 8./3
        rho = 28
        #approx_model = LorenzModel(sigma, rho, beta)
        approx_model = PeriodicApprox()
    elif data_name == 'VanDerPol':
        mistake_factor = 10
        mu = mistake_factor*0.2
        approx_model = VanDerPolModel(mu=mu, time_dependent=True)
    approx_model.eval()

    # constructing NN model that learns the dynamics
    #f = Net(hidden_dim=256, io_dim=data_dim)
    f = LorenzModel(sigma=0.9*10)
    #f = PeriodicApprox()
    #f = VanDerPolModel(mu=0.2*0.5)
    #f = KnowledgeModel(hidden_dim=50, io_dim=data_dim, known_model=approx_model)
    #f = SumKnowledgeModel(hidden_dim=50, io_dim=data_dim, known_model=approx_model)
    logging.info(f)
    params = (list(f.parameters()))

    optimizer = optim.Adam(params, lr=LR)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=False, min_lr=LR/100)

    if TRAIN_MODEL:
        if LOAD_THEN_TRAIN:
            load_models(model_dir, f)
            load_optimizer(model_dir, optimizer)

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
            shuffle=False,
            train_on_one_batch=False
        )
        trainer.train()

    else:
        #load_models(model_dir, f)
        #load_optimizer(model_dir, optimizer)
        pass

    with torch.no_grad():

        N = 1000
        test_dataset = dataset(test_data_dir, lookahead=N, tau=1, k=1, max_len=int(1.5*N))
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
        ic_state, ic_future = next(iter(test_dataloader))
        ic_state = ic_state.view(1, data_dim)
        dt_test = dt
        print(N)
        t = torch.arange(0, N*dt_test, dt_test)
        shifted_t = torch.arange(0, N*dt, dt)
        logging.info('Calculating test prediction...')
        ex_traj = np.array(odeint(f, ic_state, t).view(-1, data_dim))
        approx_traj = np.array(odeint(approx_model, ic_state, t).view(-1, data_dim))
        logging.info('Finished test prediction...')

        if data_name == 'Lorenz':
            test_lorenz(show_prior=False)

        if data_name == 'VanDerPol':
            test_van_der_pol(show_prior=True)


    logging.info("\n REACHED END OF MAIN")

