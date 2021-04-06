"""
BY: Jan-Philipp von Bassewitz
Version: 1.1

TODO:
- Dimension reduction of Hidden state with NN
- VAE?
- Loss function for Chaotic system...
- Log file!
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D  # <--- This is important for 3d plotting
import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq as torchd
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import os
import time
import logging


# CUDA
#print('CUDA is available: ', torch.cuda.is_available())
#print('CUDA decices: ', torch.cuda.get_device_name())
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
print(device)




def load_h5_data(directory):
    """
        Loads h5 files containing the Lorenz trajectories
        :returns data stored in h5 file in a dictionary with the keys being string integers
    """
    orig_trajs = []
    with h5py.File(directory, "r") as h5:
        keys = list(h5.keys())
        for key in keys:
            g = h5.get(key)
            data = np.array(g.get("data"))
            orig_trajs.append(data)
    logging.info("Successfully loaded data from {} directory and stored it in np.stack."
          " Length one Sequence: {}".format(directory, len(orig_trajs[0])))
    return np.stack(orig_trajs)


def sample_trajectories_from_orig_traj(orig_traj_stack, orig_times, sample_len, number_of_samples, type='sin'):
    """
    :return: sample trajectories from the original trajectory (number_of_samples, sample_len)
    and one time sequence of the dimension (sample_len)
    """
    sample_t_stack = []
    sample_traj_stack = []

    if type == 'lorenz':
        for n in range(number_of_samples):
            # random starting index s
            s = int(np.random.uniform(0, int(len(orig_traj_stack[0]) - sample_len)))
            stack_number = random.randint(0, len(orig_traj_stack) - 1)

            sample_t_stack.append(orig_times[s:s + sample_len])
            sample_traj_stack.append(orig_traj_stack[stack_number][s:s + sample_len])

    if type == 'sin':
        for n in range(number_of_samples):
            phase_shift = 2 * np.pi * random.random()
            sample_traj_stack.append(np.sin(orig_times[:sample_len] + phase_shift))

    return orig_times[:sample_len], np.stack(sample_traj_stack).reshape(number_of_samples, sample_len)


def save_model(save_dir, model1, model2, model3):
    if save_dir is not None:
        ckpt_path = save_dir + 'ckpt.pth'
        torch.save({
            'func_state_dict': model1.state_dict(),
            'rec_state_dict': model2.state_dict(),
            'dec_state_dict': model3.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        logging.info('Stored ckpt at {}'.format(ckpt_path))


def load_model(save_dir, model1, model2, model3):
    ckpt_path = save_dir + 'ckpt.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model1.load_state_dict(checkpoint['func_state_dict'])
        model2.load_state_dict(checkpoint['rec_state_dict'])
        model3.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info('Loaded ckpt from {}'.format(ckpt_path))


class CreationRNN(nn.Module):
    """
    GRU class for encoding and decoding
    """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, nbatch):
        super(CreationRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.nbatch = nbatch
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.o2o = nn.Linear(hidden_dim, output_dim)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.nbatch, self.hidden_dim).to(
            device)  # used to set hidden vector to zeros

    def forward(self, x, hid):
        skip = x
        x, hid = self.rnn(x, hid)
        return skip + self.o2o(x), hid


class OdeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OdeNet, self).__init__()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, t, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# ATTENTION: Only works for 1D Latent space (no Decoder)
def teacher_forcing_prediction(input_len, output_len, latent_dim, traj_stack, encoder_model, decoder_model):
    n_samples = traj_stack.size(0)
    obs_dim = 1  # only set up for observation data of dimension 1

    hid = encoder_model.init_hidden()
    for i in reversed(range(input_len)):
        x = traj_stack[:, i].view(n_samples, 1, obs_dim)
        _, hid = encoder_model.forward(x,
                                       hid)  # TODO: hid is a tensor with all the hidden states (extract the correct one!)

    traj = torch.zeros([n_samples, output_len, latent_dim])
    for i in range(output_len):
        if i == 0:
            x = torch.zeros(n_samples, 1, latent_dim)
            print(" in if : ", x.size())

        else:
            print(" in else x: ", x.size(), ' traj_stak: ', traj_stack[:, i - 1].size())
            x = traj_stack[:, i - 1].view(n_samples, 1, latent_dim)  # TODO: Maybe error here?
        x, hid = decoder_model(x, hid)
        print("x: ", x.size(), "traj: ", traj.size(), ' parted: ', traj[:, i, :].size())
        traj[:, i, :] = x.view(n_samples, latent_dim)

    return traj


def make_prediction_two_rnn(input_len, output_len, latent_dim, traj_stack, encoder_model, latent_model, decoder_model):
    # Encoding the trajectory
    n_samples = traj_stack.size(0)
    obs_dim = 1

    hid = encoder_model.init_hidden()
    for i in reversed(
            range(input_len)):  # extracting each column coming from the back, iterating through the the squence
        x = traj_stack[:, i].view(n_samples, 1, obs_dim)
        _, hid = encoder_model.forward(x, hid)

    latent_traj = torch.zeros([n_samples, output_len, latent_dim]).to(device)
    x = torch.zeros(n_samples, 1, latent_dim).to(device)
    for i in range(output_len):
        x, hid = latent_model.forward(x, hid)
        latent_traj[:, i, :] = x.view(n_samples, latent_dim)

    traj = decoder_model.forward(latent_traj)
    return traj


def make_prediction_rnn_ode(input_len, out_times, latent_dim, traj_stack, encoder_model, latent_model, decoder_model):
    # Encoding the trajectory
    n_samples = traj_stack.size(0)
    obs_dim = 1

    hid = encoder_model.init_hidden()
    for i in reversed(
            range(input_len)):  # extracting each column coming from the back, iterating through the the squence
        x = traj_stack[:, i].view(n_samples, 1, obs_dim)
        _, hid = encoder_model.forward(x, hid)

    latent_traj = odeint(latent_model, hid[-1], out_times)    # hid.size()==(n_samples, latent_dim)

    traj = decoder_model.forward(latent_traj)
    return traj.view(N_SAMPLES, len(out_times), OBS_DIM)


def make_vae_prediction_rnn_ode(input_len, out_times, latent_dim, traj_stack, encoder_model, latent_model, decoder_model):
    # Encoding the trajectory
    n_samples = traj_stack.size(0)
    obs_dim = 1

    hid = encoder_model.init_hidden()
    for i in reversed(
            range(input_len)):  # extracting each column coming from the back, iterating through the the squence
        x = traj_stack[:, i].view(n_samples, 1, obs_dim)
        _, hid = encoder_model.forward(x, hid)

    # VAE with exponential distribution!
    qz0_mean, qz0_logvar = hid[-1, :, :latent_dim], hid[-1, :, latent_dim:]
    epsilon = torch.randn(qz0_mean.size()).to(device)
    z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

    latent_traj = odeint(latent_model, z0, out_times)  # could be wrong hid

    traj = decoder_model.forward(latent_traj)
    return traj.view(N_SAMPLES, len(out_times), OBS_DIM)


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


if __name__ == "__main__":

    # Model settings
    TRAIN_MODEL = True
    LOAD_THEN_TRAIN = False
    LATENT_DIM = 20
    N_HIDDEN = 20
    NUM_LAYERS = 2
    RNN_HIDDEN = LATENT_DIM
    OBS_DIM = 1
    LEARNING_RATE = 0.001
    EPOCHS = 2000
    PATIENCE = 20

    project_dir = dir_path = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = project_dir + "/seq2seq_lorenz_prediction/data/test/data.h5"
    train_data_dir = project_dir + "/seq2seq_lorenz_prediction/data/train/data.h5"
    val_data_dir = project_dir + "/seq2seq_lorenz_prediction/data/val/data.h5"
    MODEL_DIR = project_dir + "/seq2seq_lorenz_prediction/models/PredictionRNN"
    SAVE_DIR = MODEL_DIR

    # Load Data
    ORIG_TRAJ_STACK = load_h5_data(train_data_dir)
    VAL_ORIG_TRAJ_STACK = load_h5_data(val_data_dir)
    dt = 0.1  # read out from simulation
    ORIG_TIMES = np.arange(0, 100.1, 0.1)

    # Data dimensions
    N_ORI_TRAJ = len(ORIG_TRAJ_STACK)
    SAMPLE_LEN = 50  # length of one sample trajectory
    PREDICTION_LEN = 0
    LONG_SAMPLE_LEN = SAMPLE_LEN + PREDICTION_LEN
    N_SAMPLES = 10  # number of samples
    MODEL_NAME = "LODE_EP{}_LD{}_NS{}_LSL{}".format(EPOCHS, LATENT_DIM, N_SAMPLES, LONG_SAMPLE_LEN)   # to destinguish between multiple trained models
    logging.basicConfig(filename=MODEL_NAME+'.log', level=logging.INFO, )
    FIGURES_PATH = project_dir + "/seq2seq_lorenz_prediction/figures/" + MODEL_NAME
    try:
        os.makedirs(FIGURES_PATH)
    except FileExistsError:
        logging.info('Saving figures in path that already exists.')
        pass

    # Sample trajectories from original data to train the model on
    LONG_SAMPLE_T, LONG_SAMPLE_TRAJ_STACK = sample_trajectories_from_orig_traj(ORIG_TRAJ_STACK, ORIG_TIMES,
                                                                               LONG_SAMPLE_LEN, N_SAMPLES)
    plt.plot(LONG_SAMPLE_T, LONG_SAMPLE_TRAJ_STACK[0])
    plt.title('Training data'), plt.savefig(FIGURES_PATH+'/Trainingdata.png')
    LONG_SAMPLE_T, LONG_SAMPLE_TRAJ_STACK = torch.from_numpy(LONG_SAMPLE_T).float().to(device), torch.from_numpy(
        LONG_SAMPLE_TRAJ_STACK).float().to(device)
    SAMPLE_T, SAMPLE_TRAJ_STACK = LONG_SAMPLE_T[:SAMPLE_LEN], LONG_SAMPLE_TRAJ_STACK[:, :SAMPLE_LEN]

    # create validation data
    VAL_T, VAL_TRAJ_STACK = sample_trajectories_from_orig_traj(VAL_ORIG_TRAJ_STACK, ORIG_TIMES, LONG_SAMPLE_LEN,
                                                               N_SAMPLES)
    VAL_T, VAL_TRAJ_STACK = torch.from_numpy(VAL_T).float().to(device), torch.from_numpy(VAL_TRAJ_STACK).float().to(
        device)

    # Construct NNs
    EncoderRNN = CreationRNN(input_dim=OBS_DIM, hidden_dim=RNN_HIDDEN, output_dim=LATENT_DIM, num_layers=NUM_LAYERS,
                             nbatch=N_SAMPLES).to(device)
    #LatentModel = CreationRNN(input_dim=LATENT_DIM, hidden_dim=RNN_HIDDEN, output_dim=LATENT_DIM, num_layers=NUM_LAYERS, nbatch=N_SAMPLES).to(device)
    LatentModel = OdeNet(input_dim=LATENT_DIM, hidden_dim=50, output_dim=LATENT_DIM).to(device)
    DecNet = Decoder(input_dim=LATENT_DIM, hidden_dim=N_HIDDEN, output_dim=OBS_DIM).to(device)
    logging.info('\n{} \n{} \n{}'.format(EncoderRNN, LatentModel, DecNet))

    params = (list(EncoderRNN.parameters()) + list(LatentModel.parameters()) + list(DecNet.parameters()))
    pytorch_total_params = sum(p.numel() for p in params if p.requires_grad)
    logging.info("Total number of trainable parameters: {}".format(pytorch_total_params))
    optimizer = optim.Adam(params, lr=LEARNING_RATE)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=PATIENCE, verbose=False, min_lr=LEARNING_RATE/100)

    # TRAINING THE MODEL
    if TRAIN_MODEL:
        if LOAD_THEN_TRAIN:
            load_model(SAVE_DIR, EncoderRNN, LatentModel, DecNet)

        VAL_LOSS = []
        prev_val_loss = 1e30
        TRAIN_LOSS = []
        plt.figure()
        plt.plot(LONG_SAMPLE_T.cpu(), LONG_SAMPLE_TRAJ_STACK[0, :].cpu(), 'o', label='Real Data')

        logging.info("\n Starting to train model with N_SAMPLES: {} | SAMPLE_LEN: {} | EPOCHS: {} \n".format(N_SAMPLES, SAMPLE_LEN, EPOCHS))
        pre_train_time = time.time()

        for EPOCH in tqdm(range(EPOCHS)):

            optimizer.zero_grad()
            pred_traj = make_prediction_rnn_ode(SAMPLE_LEN, LONG_SAMPLE_T, LATENT_DIM, LONG_SAMPLE_TRAJ_STACK,
                                                EncoderRNN, LatentModel, DecNet)

            # compute the loss
            LONG_SAMPLE_TRAJ_STACK = LONG_SAMPLE_TRAJ_STACK.view(N_SAMPLES, LONG_SAMPLE_LEN, OBS_DIM)
            if pred_traj.size() != LONG_SAMPLE_TRAJ_STACK[:, :].size():
                logging.info("pred_traj.size(): {}, LONG_SAMPLE_TRAJ_STACK.size(): {}".format(pred_traj.size(),
                                                                                       LONG_SAMPLE_TRAJ_STACK.size()))
                assert pred_traj.size() == LONG_SAMPLE_TRAJ_STACK[:, :].size()
            loss = torch.mean(torch.abs(pred_traj - LONG_SAMPLE_TRAJ_STACK) ** 2)
            TRAIN_LOSS.append(float(loss))
            loss.backward()
            optimizer.step()

            # compute validation loss
            VAL_TRAJ_STACK = VAL_TRAJ_STACK.view(N_SAMPLES, LONG_SAMPLE_LEN, OBS_DIM)
            val_traj = make_prediction_rnn_ode(SAMPLE_LEN, LONG_SAMPLE_T, LATENT_DIM, VAL_TRAJ_STACK, EncoderRNN,
                                               LatentModel, DecNet)
            val_loss = torch.mean(torch.abs(val_traj - VAL_TRAJ_STACK) ** 2)
            VAL_LOSS.append(float(val_loss))

            scheduler.step(val_loss)

            if EPOCH % 10 == 0:
                pass
            if EPOCH % int(4 * PATIENCE) == 0 and EPOCH != 0:
                save_model(SAVE_DIR, EncoderRNN, LatentModel, DecNet)
                logging.info("EPOCH: {} with training loss: {} | validation loss: {} | lr: {}".format(EPOCH, loss, val_loss,
                                                                                               get_lr(optimizer)))
                plt.plot(LONG_SAMPLE_T.cpu(), pred_traj[0].detach().cpu().numpy(), label="EPOCH_" + str(EPOCH))
                #if val_loss > prev_val_loss:
                #    print("Stopping training early because validation stopped improving!")
                #    break
                prev_val_loss = val_loss

        post_train_time = time.time()
        logging.info('\n Training finished after {} seconds'.format(int(post_train_time - pre_train_time)))
        plt.plot(LONG_SAMPLE_T.cpu(), pred_traj[0].detach().cpu().numpy(), label="EPOCH_" + str(EPOCH))
        plt.title('Training progress'), plt.legend(), plt.savefig(FIGURES_PATH+'/Trainingprogress.png')

        plt.figure()
        plt.plot(np.log(TRAIN_LOSS), label='Train loss'), plt.plot(np.log(VAL_LOSS), label='Validation loss')
        plt.title('Losses'), plt.legend(), plt.savefig(FIGURES_PATH+'/Losses.png')
        save_model(SAVE_DIR, EncoderRNN, LatentModel, DecNet)

    else:
        load_model(SAVE_DIR, EncoderRNN, LatentModel, DecNet)
        pass

    # TESTING THE MODEL
    TEST_LEN = SAMPLE_LEN
    EXTRAPOLATE_LEN = 200
    LONG_TEST_LEN = TEST_LEN + EXTRAPOLATE_LEN
    N_TESTS = N_SAMPLES
    ORIG_TEST_TRAJ_STACK = load_h5_data(test_data_dir)

    # Sample trajectories from original data to train the model on
    LONG_TEST_T, LONG_TEST_TRAJ_STACK = sample_trajectories_from_orig_traj(ORIG_TEST_TRAJ_STACK, ORIG_TIMES,
                                                                           LONG_TEST_LEN, N_TESTS)
    LONG_TEST_T, LONG_TEST_TRAJ_STACK = torch.from_numpy(LONG_TEST_T).float().to(device), torch.from_numpy(
        LONG_TEST_TRAJ_STACK).float().to(device)
    TEST_T, TEST_TRAJ_STACK = LONG_TEST_T[:TEST_LEN], LONG_TEST_TRAJ_STACK[:, :TEST_LEN]

    with torch.no_grad():

        pred_traj = make_prediction_rnn_ode(TEST_LEN, LONG_TEST_T, LATENT_DIM, LONG_TEST_TRAJ_STACK, EncoderRNN,
                                            LatentModel, DecNet)

        for idx in range(min(N_SAMPLES, 5)):
            plt.figure()
            plt.plot(LONG_TEST_T[:TEST_LEN].cpu(), LONG_TEST_TRAJ_STACK[idx, :TEST_LEN].cpu(), 'bo-',
                     label='Known Data (input)')
            plt.plot(LONG_TEST_T[TEST_LEN:].cpu(), LONG_TEST_TRAJ_STACK[idx, TEST_LEN:].cpu(), 'rx-',
                     label='Unknown real data')
            # plot moving average of data:
            N = 20
            MA_TRAJ = np.convolve(np.array(LONG_TEST_TRAJ_STACK[idx, :].cpu()), np.ones(N) / N, mode='valid')
            plt.plot(LONG_TEST_T[int(N / 2):-int(N / 2) + 1].cpu(), MA_TRAJ, 'k', label='MA of data')
            plt.plot(LONG_TEST_T.cpu(), pred_traj[idx].cpu(), 'g', label='Prediction')
            plt.title('Learnt behaviour'), plt.legend(), plt.savefig(FIGURES_PATH + '/' +str(idx) + 'prediction.png')

    logging.info("\n REACHED END OF MAIN")
    print("\n REACHED END OF MAIN")

