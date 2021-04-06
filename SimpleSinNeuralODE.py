"""
This was the first project of my Bachelor Thesis:

Why do this: My first implementations of Neural ODEs for the Lorentz attractor showed unexpected behaviour so I decided
to further investigate Neural ODEs by trying to solve trivial 1D ODEs.

Results: The models I trained in the end were capable of finding the dynamics of exponential, linear and quadratic functions.
This dynamics also generalized quite well for expanded domains, which is a sign of truly finding the effective dynamics.
The models had however large issues with finding the dynamics of periodic functions like sin. Even after expanding the Neural
ODEs to also get the time as an input (necessary for ODEs of the form dy/dt = f(y,t)) I was unable to recover these dynamics.
Furthermore I was not able to get good results for the right hand sides of the ODEs. This might be due to a bug in my
function that plots the learnt derivative, or to a misunderstanding on my side of how the Neural ODE learns the dynamics.

Learnings:
    - The learning rate has a very large impact on the success of the training. Different data series require different
    learning rates: To low and the model gets stuck in a local minimum; To high and it is unable to find a minimum.
    - Training on batches or even just overfitting on one particular example resulted in better or equally as good results.
    - odeint method takes y0 as the y-value at t[0] not 0!!!
    - Form of the activation function has great implications on what the model can display. So the dynamics of
    a given data series can be better found with different activation functions.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import os
from LorentzLatentODE import CreationRNN
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


# returns y values of the generative function of the ODE with IC = 0
def generative_function(type, x_value):
    if type == 'const':
        y_value = x_value
    elif type == 'squared':
        y_value = 0.5 * x_value ** 2
    elif type == 'exp':
        y_value = np.exp(x_value)
    elif type == '-const':
        y_value = -x_value
    elif type == 'sin':
        y_value = np.sin(x_value)
    elif type == 'conv':
        y_value = 1/(x_value)
    else:
        print("'generative_function' function not implemented for ", type, "type function")
        return

    return y_value


# get a random interval on the interval borders
def get_random_interval(interval):
    interval_min, interval_max = interval[0], interval[1]
    interval_len = interval_max-interval_min
    rand_start = interval_min + random.random()*interval_len/2
    rand_end = interval_max - random.random()*interval_len/2
    assert(rand_start >= interval_min and rand_end <= interval_max)
    rand_interval = [rand_start, rand_end]
    return rand_interval


# compute the function value at x[0] for different ICs of the ODE
def get_y0s(x_values, type):
    x0 = float(x_values[0])
    if type == 'const':
        y0s = x0 + np.linspace(-4, 4, 6)
    elif type == 'squared':
        y0s = 0.5 * x0 ** 2 + np.linspace(-4, 4, 6)
    elif type == 'exp':
        y0s = np.exp(x0) * np.linspace(-4, 4, 6)
    elif type == '-const':
        y0s = -x0 + np.linspace(-4, 4, 6)
    elif type== 'sin':
        y0s = np.sin(x0) + np.linspace(-4, 4, 6)
    elif type == 'conv':
        y0s = 1/(x0 + np.linspace(-4, 4, 6))
    else:
        print("'get_y0' function not implemented for ", type, "type function")
        return
    return y0s


def plot_example_prediction(type, interval, label='label'):
    data_gen = DataGenerator(type=TYPE, noise_sigma=NOISE, n_points=N_POINTS)
    data_sequence = data_gen.get_torch_batch(interval=interval)
    x_values, y_values, y0_value = data_sequence
    y_pred_value = make_prediction(data_sequence, EncoderRNN, f, DecoderNet, LATENT_DIM)
    plt.figure()
    if type == 'exp':
        plt.plot(x_values, np.log(y_values), 'o', label="Real data")
        plt.plot(x_values, np.log(np.array(y_pred_value)), label=label)
        plt.title("Example prediction (log-scale)")
    else:
        plt.plot(x_values, y_values, 'o', label="Real data")
        plt.plot(x_values, y_pred_value.detach().numpy(), label=label)
        plt.title("Example prediction")
    plt.legend()
    plt.show()


# plot learnt solutions of ODE for different initial conditions
def plot_learnt_behaviour(type, interval):
    x_values = torch.linspace(interval[0], interval[1], 20).float()
    # compute the function value at x[0] for different ICs of the ODE
    y0s = get_y0s(interval, type)

    for y0 in y0s:
        y0 = torch.from_numpy(y0.reshape(1)).float()
        y_pred_values = odeint(f, y0, x_values)
        if type=='exp' or type=='squared':
            plt.title('Learnt behaviour for different ICs (logarithmic chart)')
            plt.plot(x_values, np.log(y_pred_values))
        else:
            plt.title('Learnt behaviour for different ICs')
            plt.plot(x_values, y_pred_values)

    # Plot real derivative:
    if type == 'squared' or type == 'exp':
        plt.plot(x_values, np.log(generative_function(type, x_values)), 'x', label='Real solution for y(0)=1')
    else:
        plt.plot(x_values, generative_function(type, x_values), 'x', label='Real solution for y(0)=1')
    plt.legend()
    plt.show()


# Plot the learnt derivative three times for different state values
def plot_learnt_derivative(type, interval):
    x_values = torch.linspace(interval[0], interval[1], 10)
    y_values = torch.Tensor(get_y0s(x_values, type))
    for y_value in y_values:
        y_value = y_value.view(1)
        learnt_derivatives = []
        for x_value in x_values:
            x_value = x_value.view(1)
            learnt_derivatives.append(f.forward(x_value, y_value))
        if type == 'exp':
            plt.plot(x_values, np.log(learnt_derivatives), label=str(y_value))
        else:
            plt.plot(x_values, learnt_derivatives, label=str(y_value))
        plt.legend()

    # Plot real derivative:
    if type == 'const':
        ones = np.ones(len(x_values))
        plt.plot(x_values, ones, 'x', label='Real derivative')
    elif type == 'squared':
        plt.plot(x_values, x_values, 'x', label='Real derivative')
    elif type == 'exp':
        plt.plot(x_values, x_values, 'x', label='Real derivative')
    elif type == '-const':
        ones = np.ones(len(x_values))
        plt.plot(x_values, -ones, 'x', label='Real derivative')
    elif type == 'conv':
        plt.plot(x_values, -1/(x_values)**2) # not quite correct!
    else:
        pass
    plt.title('Learnt derivatives')
    plt.show()


class DataGenerator:
    def __init__(self, type, n_points, noise_sigma):
        self.type = type                         # data generation type
        self.n_points = n_points                 # number of data points
        self.x_values = np.array([])             # numpy array with data x values
        self.y_values = np.array([])             # numpy array with data y values
        self.noise_sigma = noise_sigma
        self.IC = 0 # initial condition of ODE at x=0 !

    # get a number of positive random x values on interval in order
    def get_random_x_values(self, interval):
        interval_start, interval_end = interval[0], interval[1]
        #stepsize = (interval_end-interval_start)/self.n_points
        self.x_values = np.linspace(interval_start, interval_end, self.n_points)
        #self.x_values = self.x_values + 0.9*stepsize*np.random.rand(self.n_points)
        self.x_values = np.array(self.x_values).reshape(self.n_points)

    # create batch of data drawn form ode
    def get_batch(self, interval):
        """
        :param n_points: number of data points
        :param type: data generation type {exponential: 'exp', linear: 'const', 2nd order: 'squared', sin:'sin', negative-linear: '-const'}
        :return: tuple of generated x and y values and the used initial condition y0
        """
        self.get_random_x_values(interval)
        self.IC = 10 * np.random.rand(1)
        noise = self.noise_sigma * np.random.rand(self.n_points)
        if self.type == 'const':
            self.y_values = self.IC + self.x_values + noise
        elif self.type == 'exp':
            self.y_values = self.IC * np.exp(self.x_values) + noise
        elif self.type == 'squared':
            self.y_values = self.IC + 0.5 * self.x_values ** 2 + noise
        elif self.type == 'sin':
            self.y_values = np.sin(self.x_values) + noise
        elif self.type == '-const':
            self.y_values = self.IC - self.x_values + noise
        elif self.type == 'conv':
            self.y_values = 1/(self.x_values + self.IC) + noise
        assert (self.x_values.shape == self.y_values.shape)

    # get a batch of data from the ode_func in torch data format
    def get_torch_batch(self, interval):
        self.get_batch(interval=interval)
        x, y = torch.from_numpy(self.x_values).float(), torch.from_numpy(self.y_values).float()
        y0 = y[0]
        x = x.view(self.n_points)
        y = y.view(self.n_points)
        assert (x.shape == y.shape)
        return x, y, y0.view(1)

    # returns a list of data batches
    def get_torch_batches(self, interval, n_seqences):
        data_series_batches = []
        for j in range(n_seqences):
            data_series_batches.append(self.get_torch_batch(interval))
        return data_series_batches


# build the Neural ODE model
class Net(nn.Module):
    """
        The NN that learns the right hand side of the ODE
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()       # Run init of parent class
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=True),
        )

        for m in self.net.modules():
            # iterates through the defined layers
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)      # assigns weights drawn from normal distribution
                nn.init.constant_(m.bias, val=0)                # assigns bias to layer

    def forward(self, t, x):
        return self.net(x)

# TODO: experiment with different arcitectures and activation functions to make sin possible!
# build Neural ODE model with time dependence
class TimeNet(nn.Module):
    """
    This NN also takes time dependence into account
    """
    def __init__(self, hidden_dim=50):
        super(TimeNet, self).__init__()
        self.input_dim = 2
        self.output_dim = 1
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.Softplus(),
            nn.Linear(hidden_dim, self.output_dim, bias=True),
        )

        for m in self.net.modules():
            # iterates through the defined layers
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)      # assigns weights drawn from normal distribution
                nn.init.constant_(m.bias, val=0)                # assigns bias to layer

    def forward(self, t, x):
        t = t.view(1)
        vec = torch.cat((t, x))
        return self.net(vec)



class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, output_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def save_model(save_dir, encoder, odenet, decoder):
    torch.save({
       'enc_state_dict': encoder.state_dict(),
        'ode_state_dict': odenet.state_dict(),
        'dec_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_dir)
    print('Stored ckpt at {}'.format(save_dir))


def load_model(save_dir, encoder, odenet, decoder):
    if os.path.exists(save_dir):
        checkpoint = torch.load(save_dir)
        encoder.load_state_dict(checkpoint['enc_state_dict'])
        odenet.load_state_dict(checkpoint['ode_state_dict'])
        decoder.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loaded ckpt from {}'.format(save_dir))


def make_prediction(data_tuple, encoder_model, ode_net, decoder_model, latent_dim):
    x, y, y0 = data_tuple
    input_len = len(y)
    output_len = len(x)

    # perform the encoding -> last hidden state is context vector used as the inital condition for Neural ODE
    hid = encoder_model.init_hidden()
    #print("hid: {}".format(hid.size()))
    for idx in reversed(range(input_len)):
        inp = y[idx].view(1, 1, 1)
        _, hid = encoder_model.forward(inp, hid)


    latent_y_pred = torch.zeros([latent_dim, output_len])
    #print("hid.size(): ", hid.size())
    #print("latent_y_pred.size() :", latent_y_pred.size())

    #hid = hid[1].view(latent_dim)
    #latent_y_pred = odeint(ode_net, hid, x)

    x = torch.zeros(1, 1, latent_dim)
    for idx in range(output_len):
        x, hid = ode_net.forward(x, hid)
        #print("x.size():", x.size())
        latent_y_pred[:, idx] = x.view(latent_dim)

    #print("latent_y_pred.size() :", latent_y_pred.size())
    latent_y_pred = torch.transpose(latent_y_pred, 0, 1)

    y_pred = decoder_model.forward(latent_y_pred)

    return y_pred


if __name__ == '__main__':

    # Settings
    TRAIN_MODEL = True
    SAVE_MODEL = True
    LOAD_AND_TRAIN = False
    LOG_PROGRESS = False
    EPOCHS = 100
    LR = 0.01
    N_POINTS = 10
    N_SEQUENCES = 10
    NOISE = 0
    TYPE = 'const'    # allowed: exp, sin, squared, const
    LATENT_DIM = 6
    VAL_INTERVAL = [1, 15]
    TRAIN_INTERVAL = [1, 15]
    TEST_INTERVAL = [1, 15]

    FOLDER_PATH = "first_tries/SimpleModels/"
    FILENAMES = {'exp': 'simplemodel.pth.tar', 'sin': 'sinmodel.pth.tar', 'squared': 'squaredmodel.pth.tar', 'const': 'constmodel.pth.tar', '-const': '-constmodel.pth.tar', 'conv': 'convmodel.pth.tar'}
    FILENAME = FOLDER_PATH + FILENAMES[TYPE]
    TRAIN_LOSS = []
    VAL_LOSS = []

    # Construct the GRU Encoder
    EncoderRNN = CreationRNN(input_dim=1, hidden_dim=LATENT_DIM, num_layers=2, output_dim=LATENT_DIM, nbatch=1)
    print(EncoderRNN)

    # Construct the ODENet
    # f = Net(input_dim=LATENT_DIM, hidden_dim=LATENT_DIM, output_dim=LATENT_DIM)
    f = CreationRNN(input_dim=LATENT_DIM, hidden_dim=LATENT_DIM, num_layers=2, output_dim=LATENT_DIM, nbatch=1)
    print(f)

    # Construct the Decoder
    DecoderNet = Decoder(input_dim=LATENT_DIM, n_hidden=LATENT_DIM, output_dim=1)
    print(DecoderNet)

    params = (list(f.parameters()) + list(EncoderRNN.parameters()) + list(DecoderNet.parameters()))
    optimizer = optim.AdamW(params, lr=LR)  # be very careful with the learning rate!
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    if TRAIN_MODEL:
        if LOAD_AND_TRAIN:
            load_model(FILENAME, EncoderRNN, f, DecoderNet)
        with torch.no_grad():
            plot_example_prediction(type=TYPE, label='Prior to training', interval=VAL_INTERVAL)
            pass

        data_generator = DataGenerator(type=TYPE, noise_sigma=NOISE, n_points=N_POINTS)
        batches = data_generator.get_torch_batches(interval=TRAIN_INTERVAL, n_seqences=N_SEQUENCES)
        #print("Example Sequence: ", batches[0])

        for EPOCH in tqdm(range(EPOCHS)):

            for sequence in batches:
                optimizer.zero_grad()

                x_train, y_train, y0_train = sequence
                #print("x: {} y: {} y0: {}".format(x_train.size(), y_train.size(), y0_train.size()))

                # prediction with Neural ODE
                y_pred = make_prediction(sequence, EncoderRNN, f, DecoderNet, LATENT_DIM)
                #print("y_pred: {} y_train: {}".format(y_pred.size(), y_train.size()))

                loss = torch.mean(torch.abs(y_pred-y_train))
                TRAIN_LOSS.append(float(loss))

                loss.backward()
                optimizer.step()
                scheduler.step()

                # compute validation loss every step
                val_sequence = data_generator.get_torch_batch(interval=get_random_interval(VAL_INTERVAL))
                _, y_val, _ = val_sequence
                y_val_pred = make_prediction(val_sequence, EncoderRNN, f, DecoderNet, LATENT_DIM)
                y_val = y_val.view(N_POINTS, 1)
                val_loss = torch.mean(torch.abs(y_val - y_val_pred))
                VAL_LOSS.append(val_loss)

            print("loss on training data: {} after epoch {}".format(float(loss), EPOCH+1))
            if SAVE_MODEL and EPOCH % 25 == 0:
                save_model(FILENAME, EncoderRNN, f, DecoderNet)
            if LOG_PROGRESS:
                with torch.no_grad():
                    plot_example_prediction(type=TYPE, label="On epoch: "+str(EPOCH+1), interval=VAL_INTERVAL)

        print("\n ---Training is finished!--- \n")
        plot_example_prediction(type=TYPE, label='After training', interval=VAL_INTERVAL)

        if SAVE_MODEL:
            save_model(FILENAME, EncoderRNN, f, DecoderNet)
        plt.plot(np.log(TRAIN_LOSS), label='TRAIN_LOSS')
        plt.plot(np.log(VAL_LOSS), label='VAL_LOSS')
        plt.legend(), plt.show()
    else:
        load_model(FILENAME, EncoderRNN, f, DecoderNet)

    # testing model performance
    with torch.no_grad():
        plot_example_prediction(type=TYPE, label='After training', interval=TRAIN_INTERVAL)
        for i in range(2):
            plot_example_prediction(type=TYPE, label='After training', interval=TEST_INTERVAL)
        #plot_learnt_derivative(type=TYPE, interval=TEST_INTERVAL)
        #plot_learnt_behaviour(type=TYPE, interval=TEST_INTERVAL)
