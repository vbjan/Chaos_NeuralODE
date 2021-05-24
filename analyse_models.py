"""
    Created by: Jan-Philipp von Bassewitz, CSE-lab, ETH Zurich

    Analysis of trained added knowledge based models of Lorenz system
"""
import numpy as np
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
from utils import save_models, load_models, get_num_trainable_params
from Datasets import DDDLorenzData, VanDerPol
from threeDLorenzNODE import SumKnowledgeModel, Net, KnowledgeModel, LorenzModel, PeriodicApprox


class GhostModel(nn.Module):
    def __init__(self, io_dim=3):
        super().__init__()       # Run init of parent class
        self.temp = nn.Linear(1, 1)
        self.io_dim = io_dim

    def forward(self, t, x):
        return 0*x


def x_of_exact_lorenz(x):
    sigma = 10
    return sigma*(x[:, 1] - x[:, 0])


if __name__ == "__main__":
    # set up matplotlib font size
    font = {'size': 18}
    matplotlib.rc('font', **font)

    type = "MNODE"
    #type = "AddNODE"
    #type = "pure"

    dt = 0.01  # read out from simulation script
    project_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = project_dir + "/data/Data3D" + str(dt) + "/test/data.h5"
    fig_dir = project_dir + '/3D_lorenz_prediction/figures/'

    N = 2
    batch_size = 1000
    dataset = DDDLorenzData(test_data_dir, lookahead=N, tau=1, k=1, max_len=int(batch_size * 1.5))
    data_dim = dataset.data_dim
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    # TODO: change prior models to Lorenz model to make comparison better
    if type == 'AddNODE':    # knowledge based model
        mistake_factor = 0.6
        label = type + str(mistake_factor)
        ghost = LorenzModel(sigma=mistake_factor*10)
        f = SumKnowledgeModel(hidden_dim=50, known_model=ghost, io_dim=data_dim)
        model_dir = project_dir + '/3D_lorenz_prediction/models/sum_models/' + str(mistake_factor) + 'model/knowledge'
    elif type == 'MNODE':
        mistake_factor = 0.9
        label = type + str(mistake_factor)
        #label = type + 'periodic'
        ghost = LorenzModel(sigma=mistake_factor*10)
        #ghost = PeriodicApprox()
        f = KnowledgeModel(hidden_dim=50, known_model=ghost, io_dim=data_dim)
        model_dir = project_dir + '/3D_lorenz_prediction/models/knowledgemodels/' + str(mistake_factor) + 'sigma/knowledge'
        #model_dir = project_dir + '/3D_lorenz_prediction/models/knowledgemodels/periodic_model/knowledge'
    elif type == 'pure':
        label = "pure"
        f = Net(hidden_dim=256, io_dim=data_dim)
        model_dir = project_dir + '/3D_lorenz_prediction/models/2.4model/3DLorenzmodel'
    else:
        print("type '{}' doesn't exist".format(type))
        exit()
    load_models(model_dir, f)

    ic_state, ic_future = next(iter(dataloader))
    ic_state = ic_state.view(batch_size, data_dim)

    #vec = torch.tensor([[2, 10, 2]]).float()
    t = torch.zeros(1)
    dstatedt = f(t, ic_state)
    dxdt_NODE = dstatedt[:, 0].detach().numpy()
    if type == 'pure':
        dxdt_NODE = dxdt_NODE/10
    dxdt = x_of_exact_lorenz(ic_state).detach().numpy()

    comp = np.arange(-10, 10)
    plt.plot(dxdt, dxdt_NODE, 'x', color='orange')
    plt.xlabel(r'$(1-\alpha)\sigma(y-x)$')
    plt.ylabel('Learnt $dx/dt$')
    plt.savefig(fig_dir + label + 'lorenz_error_cobweb.png',
                dpi=300,
                format='png',
                bbox_inches='tight')
    plt.show()

    #error = dxdt_NODE/dxdt
    error = (dxdt_NODE-dxdt)#/(1-mistake_factor)
    mean_error = np.mean(np.abs(error))
    print("\n Mean error: {}".format(mean_error))

    x = ic_state[:, 0]
    y = ic_state[:, 1]

    # DEVIL
    devil_state = (torch.rand(ic_state.size()).float()-0.5)*45 # random inputs
    devil_dxdt_NODE = f(t, devil_state)[:, 0].detach().numpy()
    if type == 'pure':
        devil_dxdt_NODE = devil_dxdt_NODE/10
    devil_dxdt = x_of_exact_lorenz(devil_state).detach().numpy()

    #devil_error = devil_dxdt_NODE/devil_dxdt
    devil_error = (devil_dxdt_NODE - devil_dxdt)#/(1-mistake_factor)
    print(devil_error.shape)

    nbins = 50
    plt.hist(devil_error, bins=nbins, label='Random inputs')
    plt.hist(error, bins=int(nbins/5), label='Lorenz inputs')
    #plt.title('Histogram of error')
    plt.ylabel('# of observations')
    plt.xlabel('Error')
    #plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=1)
    plt.savefig(fig_dir + label + 'error_histogram.png',
                dpi=300,
                format='png',
                bbox_inches='tight')
    plt.show()

    devil_error = np.abs(devil_error)
    plt.plot(devil_dxdt, devil_dxdt_NODE, 'x')
    plt.xlabel(r'$(1-\alpha)\sigma(y-x)$')
    plt.ylabel(r'Learnt $dx/dt$')
    plt.savefig(fig_dir + label + 'random_error_cobweb.png',
                dpi=300,
                format='png',
                bbox_inches='tight')
    #plt.title('Error on non-Lorenz inputs')
    plt.show()

    devil_x = devil_state[:, 0].detach().numpy()
    devil_y = devil_state[:, 1].detach().numpy()

    print('Mean error on non-Lorenz inputs', np.mean(devil_error))

    f, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    im = ax.tricontourf(devil_x, devil_y, devil_error, 300)  # choose 20 contour levels, just to show how good its interpolation is
    f.colorbar(im, label=r'Absolute error of learnt $dx/dt$')
    im.set_clim(0, 100)
    plt.plot(x, y, 'rx', label='Lorenz trajectory')

    plt.xlabel('x')
    plt.ylabel('y')
    lg = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=1)
    plt.savefig(fig_dir + label + 'error_location.png',
                dpi=300,
                format='png',
                bbox_inches='tight')
    plt.show()


    # TODO: Heat map of where the error is...



    #plt.plot(traj[:, 0].detach().numpy())
    #plt.plot(traj[:, 1].detach().numpy())
    #plt.plot(traj[:, 2].detach().numpy())
