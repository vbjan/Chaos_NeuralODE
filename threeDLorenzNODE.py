'''
Version: 6.1
TODO:
- delete xi return of data functions?
- write so that logging.info appears in console as well
- save checkpoints of good models better!
- Use all the steps for training
'''

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
from utils import save_models, load_models, load_h5_data, z1test, split_sequence, make_batches_from_stack

os.environ['KMP_DUPLICATE_LIB_OK']='True'
logging.basicConfig(filename="3D_lorenz_prediction/3DLorenzNODE.log", level=logging.INFO,
                    format='%(asctime)s:%(funcName)s:%(levelname)s:%(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


def plot_lorenz(x1, x2, x3, end=100, label="plot"):
    ax = plt.axes(projection='3d')
    ax.plot3D(x1[:end], x2[:end], x3[:end], '-', label=label)
    ax.legend()
    #plt.show()


def get_batch(d, batch_n, t, batch_length=100):
    true_X = d[batch_n]
    allowed_numbers = len(true_X)-batch_length
    s = int(np.random.uniform(0, allowed_numbers))
    batch_X = true_X[s:s+batch_length, :]
    batch_X0 = true_X[s]
    batch_t = t[s:s+batch_length]
    return batch_X, batch_X0, batch_t


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
    logging.info("\n ----------------------- Starting of new script ----------------------- \n")

    project_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = project_dir + "/3D_lorenz_prediction/Data3D/test/data.h5"
    train_data_dir = project_dir + "/3D_lorenz_prediction/Data3D/train/data.h5"
    val_data_dir = project_dir + "/3D_lorenz_prediction/Data3D/val/data.h5"
    figures_dir = project_dir + "/3D_lorenz_prediction/figures"
    model_dir = project_dir + '/3D_lorenz_prediction/models/3DLorenzmodel'

    # Load in the data
    d_train = load_h5_data(train_data_dir)
    logging.info("Shape of d_train: {}".format(d_train.shape))
    d_val = load_h5_data(val_data_dir)
    d_test = load_h5_data(test_data_dir)
    n_of_data = len(d_train[1])
    dt = 0.0025    # read out from simulation script
    lookahead = 2
    n_of_batches = 4
    t = torch.from_numpy(np.arange(0, (1+lookahead) * dt, dt))

    # Settings
    TRAIN_MODEL = False
    LOAD_THEN_TRAIN = False
    EPOCHS = 2000
    LR = 0.01

    # Construct model
    f = Net(hidden_dim=256)
    logging.info(f)

    params = list(f.parameters())
    optimizer = optim.Adam(params, lr=LR)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=False, min_lr=LR/100)

    if TRAIN_MODEL:
        if LOAD_THEN_TRAIN:
            load_models(model_dir, optimizer, f)

        batches_X0, batches_X = make_batches_from_stack(
                                                    d_train,
                                                    lookahead,
                                                    tau=1,
                                                    k=1,
                                                    n_batches=n_of_batches,
                                                    max_blocks=1900
                                                    )

        val_X0, val_X = make_batches_from_stack(d_val,
                                                lookahead,
                                                tau=1,
                                                k=1,
                                                n_batches=1
                                                )

        val_X0, val_X = val_X0[0].view(-1, 3), val_X[0]

        val_losses = []
        train_losses = []
        pre_val_loss = 1000000

        logging.info("\nSTARTING TO TRAIN MODEL | EPOCHS : {} | lr: {} | n_of_batches: {} | lookahead: {}"
                     .format(EPOCHS, LR, n_of_batches, lookahead))

        pre_train_time = time.time()
        for EPOCH in range(EPOCHS):

            for n in range(n_of_batches):
                batch_X0, batch_X = batches_X0[n].view(-1, 3), batches_X[n]

                optimizer.zero_grad()
                X_pred = odeint(f, batch_X0, t).permute(1, 0, 2)

                loss = torch.mean(torch.abs(X_pred - batch_X) ** 2)
                loss.backward()
                optimizer.step()

            train_losses.append(float(loss))
            X_val_pred = odeint(f, val_X0, t).permute(1, 0, 2)
            val_loss = torch.mean(torch.abs(X_val_pred - val_X) ** 2)
            val_losses.append(float(val_loss))
            scheduler.step(val_loss)

            if EPOCH % 10 == 0:
                logging.info("EPOCH {} finished with training loss: {} | validation loss: {} | lr: {} | K: {} \n"
                      .format(EPOCH, loss, val_loss, get_lr(optimizer), get_approx_k_of_model(f, d_val)))
                save_models(model_dir, optimizer, f)
                if val_loss > pre_val_loss and EPOCH % 30 == 0:
                    logging.info("\n STOPPING TRAINING EARLY BECAUSE VAL.LOSS STOPPED IMPROVING!\n")
                    break
            pre_val_loss = val_loss

        post_train_time = time.time()
        logging.info('\nTRAINING FINISHED after {} seconds'.format(int(post_train_time - pre_train_time)))
        plt.plot(train_losses, label='train loss')
        plt.plot(val_losses, label='validation loss')
        plt.legend(), plt.savefig(figures_dir + '/losses.png'), plt.show()
    else:
        load_models(model_dir, optimizer, f)

    with torch.no_grad():
        idx = 8
        ic_state = torch.from_numpy(d_test[idx][0, :]).float().view(1, 3)
        dt_test = 0.0025
        t = torch.arange(0, 1.5, dt_test)
        N = len(t)
        #t = t + 0.9*dt_test*np.random.rand(N)
        ex_traj = np.array(odeint(f, ic_state, t).view(-1, 3))

        # example plot of data
        end = 200
        x = d_test[idx][:, 0].reshape(-1)
        y = d_test[idx][:, 1].reshape(-1)
        z = d_test[idx][:, 2].reshape(-1)
        ax = plt.axes(projection='3d')
        ax.plot3D(x[:end], y[:end], z[:end], '-', label='Real Lorenz')
        ax.plot3D(ex_traj[:, 0], ex_traj[:, 1], ex_traj[:, 2], '-', label="Learnt Lorenz")
        ax.legend()
        plt.savefig(figures_dir + "/lorenz3d.png")
        plt.show()


        # Compare x, y and z of real and learnt trajectory
        plt.figure()
        plt.plot(x[:N], label='Real x')
        plt.plot(ex_traj[:, 0], label='Learnt x')
        plt.legend()
        plt.savefig(figures_dir + "/lorenzx.png")
        plt.show()

        plt.figure()
        plt.plot(y[:N], label='Real y')
        plt.plot(ex_traj[:, 1], label='Learnt y')
        plt.legend()
        plt.savefig(figures_dir + "/lorenzy.png")
        plt.show()

        plt.figure()
        plt.plot(z[:N], label='Real z')
        plt.plot(ex_traj[:, 2], label='Learnt z')
        plt.legend()
        plt.savefig(figures_dir + "/lorenzz.png")
        plt.show()

        # 0-1 Test for chaos in real and learnt trajectories
        logging.info("0-1 test of | real x: {} | learnt x: {}".format(z1test(x[:N]), z1test(ex_traj[:, 0])))
        logging.info("            | real y: {} | learnt y: {}".format(z1test(y[:N]), z1test(ex_traj[:, 1])))
        logging.info("            | real z: {} | learnt z: {}".format(z1test(z[:N]), z1test(ex_traj[:, 2])))


    logging.info("\n REACHED END OF MAIN")





'''
def recast_sequence_to_batches(sequence, lookah):
    """
    :param sequence: numpy array with trajectory
    :param lookah: lookahead for input output split
    :return: np array with all the states and np stack with the resulting trajectories
    """
    N = len(sequence)

    x0s = sequence[0::lookah + 1]
    xis = []
    xs = []
    for i in range(N):
        if i % (lookah+1) == 0:
            xis.append(sequence[i+1:i+lookah+1:1])
            xs.append(sequence[i:i+lookah+1:1])
    # Cut last x0 if sequence was not long enough
    xis = np.stack(xis[:-1])
    xs = np.stack(xs[:-1])
    if x0s.shape[0] != xis.shape[0]:
        x0s = x0s[:-1]
    #logging.info(len(x0s), len(xis), len(xs))
    assert len(x0s) == len(xis) and len(x0s) == len(xs)
    return x0s, xis, xs  # x0s.dim=(n_batches,3), xis.dim=(n_batches, lookah, 3)
'''

'''
def make_trainable_data_from_sequence(sequence, lookah):
    """
    :param sequence: data sequence to make data from
    :param lookah: lookahead parameter for data splitting
    :return: x0, xi and x as torch tensors of size (n_batches, 1 or 1+lookah, 3)
    """
    x0, xi, x = recast_sequence_to_batches(sequence, lookah)
    x0 = torch.from_numpy(x0).float()
    xi = torch.from_numpy(xi).float()
    x = torch.from_numpy(x).float()
    return x0, xi, x
'''


'''
def make_trainable_batches_from_stack(stack, lookah, n_batches):
    batches_xs = []
    batches_xis = []
    batches_x0s = []
    for i in range(n_batches):
        sequence = stack[i][:, :]
        x0s, xis, xs = make_trainable_data_from_sequence(sequence, lookah)
        batches_xs.append(xs)
        batches_x0s.append(x0s)
        batches_xis.append(xis)

    return batches_x0s, batches_xis, batches_xs
'''

