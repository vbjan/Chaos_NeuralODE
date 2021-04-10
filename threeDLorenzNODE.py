'''

    Version: 10 BIG
    TODO:
        - save checkpoints of good models better!
        - Try out different dts (what is the best dt?!) for training
        - ANODE is extremely unstable!! change dimension?

    Learnings:
        - ANODE becomes unstable for chaotic systems. Penalty by putting augmented dimensions in loss function,
        creates periodic behavior. -> Solution (?): Use Sigmoid as activation function of augmented states

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
from utils import save_models, load_models, save_optimizer, load_optimizer
from Datasets import DDDLorenzData


def plot_lorenz(x1, x2, x3, end=100, label="plot"):
    ax = plt.axes(projection='3d')
    ax.plot3D(x1[:end], x2[:end], x3[:end], '-', label=label)
    ax.legend()
    #plt.show()


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


class ANODE(nn.Module):
    """
    Augmented Neural ODE
    """
    def __init__(self, aug_dim, hidden_dim=100):
        super().__init__()  # Run init of parent class
        self.io_dim = 3 + aug_dim
        #self.acti = nn.Tanh()
        self.acti = nn.LeakyReLU()
        self.layer1 = nn.Linear(self.io_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, self.io_dim)

    def forward(self, t, x):
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


if __name__ == "__main__":

    # logging settings
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    logging.basicConfig(filename="3D_lorenz_prediction/ANODE.log", level=logging.INFO,
                        format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    logging.info("\n ----------------------- Starting of new script ----------------------- \n")

    # directory settings
    project_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = project_dir + "/3D_lorenz_prediction/Data3D0.01/test/data.h5"
    train_data_dir = project_dir + "/3D_lorenz_prediction/Data3D0.01/train/data.h5"
    val_data_dir = project_dir + "/3D_lorenz_prediction/Data3D0.01/val/data.h5"
    figures_dir = project_dir + "/3D_lorenz_prediction/figures"
    model_dir = project_dir + '/3D_lorenz_prediction/models/3DLorenzmodel'
    #model_dir = project_dir + '/3D_lorenz_prediction/models/ANODE'

    # data settings
    dt = 0.01    # read out from simulation script
    lookahead = 2
    batch_size = 200
    t = torch.from_numpy(np.arange(0, (1+lookahead) * dt, dt))
    n_groups = 2

    # model settings
    TRAIN_MODEL = True
    LOAD_THEN_TRAIN = False
    EPOCHS = 1000
    LR = 0.01
    augmentation_dim = 0

    # construct model
    f = Net(hidden_dim=500)
    #f = ANODE(aug_dim=augmentation_dim, hidden_dim=256)
    logging.info(f)

    params = (list(f.parameters()))
    optimizer = optim.Adam(params, lr=LR)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=False, min_lr=LR/100)

    if TRAIN_MODEL:
        if LOAD_THEN_TRAIN:
            load_models(model_dir, f)
            load_optimizer(model_dir, optimizer)

        train_dataset = DDDLorenzData(train_data_dir, lookahead, tau=1, k=1)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, n_groups=n_groups)

        val_dataset = DDDLorenzData(val_data_dir, lookahead, tau=1, k=1)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        val_losses = []
        train_losses = []
        pre_val_loss = 1000000

        n_iterations = int(len(train_dataset)/batch_size)
        logging.info("\nSTARTING TO TRAIN MODEL | EPOCHS : {} | lr: {} | batchsize: {} | lookahead: {}"
                     .format(EPOCHS, LR, batch_size, lookahead))
        logging.info("                          | augmented dimension: {} | iterations: {} \n"
                                                .format(augmentation_dim, n_iterations))

        pre_train_time = time.time()
        for EPOCH in range(EPOCHS):

            for n, (batch_X0, batch_X) in enumerate(train_dataloader):

                optimizer.zero_grad()
                batch_X0 = batch_X0.view(batch_size, 3)
                # build ANODE -> concat batch_X0 (blocks, x_dim) with 0s
                # concat along dim=1
                batch_a = torch.zeros(batch_size, augmentation_dim)
                augm_X0 = torch.cat((batch_X0, batch_a), dim=1)
                augm_X_pred = odeint(f, augm_X0, t).permute(1, 0, 2)
                X_pred = augm_X_pred[:, :, :3]
                loss = torch.mean(torch.abs(X_pred - batch_X) ** 2) + torch.mean(augm_X_pred[:, :, 3:]**2)
                loss.backward()
                optimizer.step()

            train_losses.append(float(loss.detach()))

            val_X0, val_X = next(iter(val_dataloader))
            val_X0 = val_X0.view(batch_size, 3)
            val_batch_a = torch.zeros(batch_size, augmentation_dim)
            augm_val_X0 = torch.cat((val_X0, val_batch_a), dim=1).detach()
            augm_X_val_pred = odeint(f, augm_val_X0, t.detach()).permute(1, 0, 2)
            X_val_pred = augm_X_val_pred[:, :, 0:3]
            val_loss = torch.mean(torch.abs(X_val_pred - val_X) ** 2)
            val_losses.append(float(val_loss))

            scheduler.step(val_loss)

            if EPOCH % 5 == 0:
                logging.info("EPOCH {} finished with training loss: {} | validation loss: {} | lr: {} "
                      .format(EPOCH, loss, val_loss, get_lr(optimizer)))
                save_models(model_dir, f)
                save_optimizer(model_dir, optimizer)
                if val_loss > pre_val_loss and EPOCH % 20 == 0:
                    logging.info("\n STOPPING TRAINING EARLY BECAUSE VAL.LOSS STOPPED IMPROVING!\n")
                    break
            pre_val_loss = val_loss

        post_train_time = time.time()
        logging.info('\nTRAINING FINISHED after {} seconds'.format(int(post_train_time - pre_train_time)))
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='validation loss')
        plt.legend(), plt.savefig(figures_dir + '/losses.png'), plt.show()
    else:
        load_models(model_dir, f)
        load_optimizer(model_dir, optimizer)

    with torch.no_grad():

        N = 50
        test_dataset = DDDLorenzData(test_data_dir, lookahead=N, tau=1, k=1)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
        ic_state, ic_future = next(iter(test_dataloader))
        ic_state = ic_state.view(1, 3)
        augm_ic_state = torch.cat((ic_state, torch.zeros(1, augmentation_dim)), dim=1)
        dt_test = 0.002
        N_pred = int(N*dt/dt_test)
        t = torch.arange(0, N_pred*dt_test, dt_test)
        augm_ex_traj = np.array(odeint(f, augm_ic_state, t).view(-1, 3 + augmentation_dim))
        ex_traj = augm_ex_traj[:, :3]

        # example plot of data
        x = ic_future[:, :, 0].view(-1)
        y = ic_future[:, :, 1].view(-1)
        z = ic_future[:, :, 2].view(-1)
        ax = plt.axes(projection='3d')
        ax.plot3D(x, y, z, '-', label='Real Lorenz')
        ax.plot3D(ex_traj[:, 0], ex_traj[:, 1], ex_traj[:, 2], '-', label="Learnt Lorenz")
        ax.legend()
        plt.savefig(figures_dir + "/lorenz3d.png")
        plt.show()

        data_time = np.arange(0, (N+1)*dt, dt)
        # compare x, y and z of real and learnt trajectory
        plt.figure()
        plt.plot(data_time, x, label='Real x')
        plt.plot(t, ex_traj[:, 0], label='Learnt x')
        plt.xlabel('time t')
        plt.legend()
        plt.savefig(figures_dir + "/lorenzx.png")
        plt.show()

        plt.figure()
        plt.plot(data_time, y, label='Real y')
        plt.plot(t, ex_traj[:, 1], label='Learnt y')
        plt.xlabel('time t')
        plt.legend()
        plt.savefig(figures_dir + "/lorenzy.png")
        plt.show()

        plt.figure()
        plt.plot(data_time, z, label='Real z')
        plt.plot(t, ex_traj[:, 2], label='Learnt z')
        plt.xlabel('time t')
        plt.legend()
        plt.savefig(figures_dir + "/lorenzz.png")
        plt.show()

        # 0-1 Test for chaos in real and learnt trajectories
        logging.info("0-1 test of | real x: {} | learnt x: {}".format(z1test(x.numpy()), z1test(ex_traj[:, 0])))
        logging.info("            | real y: {} | learnt y: {}".format(z1test(y.numpy()), z1test(ex_traj[:, 1])))
        logging.info("            | real z: {} | learnt z: {}".format(z1test(z.numpy()), z1test(ex_traj[:, 2])))


    logging.info("\n REACHED END OF MAIN")
