# -*- coding: utf-8 -*-
"""Ex1 diffusion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wrVn_gdJtYZIdkKB1y-1W1QP2OAislED

# Imports
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from google.colab import drive
drive.mount('/content/drive')

"""# Constants"""

NAM_LABELS = 5
NON_CONDITIONAL = "NON_CONDITIONAL"
CONDITIONAL = "CONDITIONAL"
EXP_SCHEDULER = "EXP"
COS_SCHEDULER = "COS"
POL_SCHEDULER = "POL"
NUM_EPOCHS = 10000
TRAIN_DATASET_SIZE = 10000
RANDOM_SEED = 0
NUM_SAMPLES = 1000
T = 1000

"""# Define desoiser classes"""

class Denoiser(nn.Module):
    # A simple feedforward network with two hidden layers
    def __init__(self, dim=64):
        super().__init__()
        self.type = NON_CONDITIONAL
        self.dim = dim
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(3, dim)  # input is (x, y, t)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, 2)  # output is (epsilon_x, epsilon_y)

    def forward(self, x, t):
        # x is a tensor of shape (batch_size, 2) containing (x, y)
        # t is a tensor (batch_size, 1)
        h = torch.cat([x, t], dim=-1)
        h = self.leaky_relu(self.fc1(h))
        h = self.leaky_relu(self.fc2(h))
        out = self.fc3(h)
        return out

class ConDenoiser(nn.Module):
    def __init__(self, dim=64, embed_dim=5):
        super().__init__()
        self.type = NON_CONDITIONAL
        self.dim = dim
        self.label_embed = nn.Embedding(NAM_LABELS, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(3 + embed_dim, dim)  # input is (x, y, t, embed(c))
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, 2)  # output is (epsilon_x, epsilon_y)

    def forward(self, x, t, c):
        c_embed = self.label_embed(c)
        h = torch.cat([x, t, c_embed], dim=-1)
        h = self.leaky_relu(self.fc1(h))
        h = self.leaky_relu(self.fc2(h))
        out = self.fc3(h)
        return out

"""# Visualisztion methods"""

def visualize_points(points, lim=2):
    pnts = points.detach().numpy()
    plt.scatter(pnts[:, 0], pnts[:, 1], color='blue', s=2)
    # Plot the square as a red rectangle
    plt.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], color='red')
    # Set the axis limits and labels
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel('x')
    plt.ylabel('y')
    # Show the plot
    plt.show()

def visualize_points_con(x0, labels, lim=2, show=True):
    import matplotlib.colors  # import this module to use ListedColormap

    # get the x and y coordinates of the points
    x = x0[:, 0].detach().numpy()
    y = x0[:, 1].detach().numpy()
    # create a list of colors for each label
    colors = ["red", "green", "blue", "black", "purple"]
    # create a scatter plot of the points with different colors for each label
    plt.scatter(x, y, c=labels.squeeze(), cmap=matplotlib.colors.ListedColormap(colors), s=5)
    plt.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], color='red')
    # Set the axis limits and labels
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    # add some labels and title to the plot
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Points and labels")

    # show the plot
    print(f"visualize_points_con: {show}")
    if show:
      plt.show()

def plot_denoiser_loss(losses):
    epochs = range(1, len(losses) + 1)
    # Plot the loss as a function of the epochs
    plt.plot(epochs, losses)
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    # Show the plot
    plt.show()


def plot_trajectory(dt_s, trajectories, type, lim=2):
    """
    plot the trajectory of 2D points
    :param dt_s: and ndarray of size (number of steps,)
    :param trajectories: a list of trajectories, each trajectory is a list of (x, y) tuples, or just 0ne trajectory
    :param type: if NON_CONDITIONAL : trajectories is only one trajectory, and if CONDITIONAL then trajectories is a
                list of trajectories
    """
    if type == NON_CONDITIONAL:
        plt.scatter(*zip(*trajectories), c=dt_s)
    else:
        cmap = plt.get_cmap('jet')
        # Scale the index of each trajectory to [0, 1] interval
        n = len(trajectories)
        inds = [3,0,2,1,4]
        colors = ["red", "green", "blue", "black", "purple"]
        for i, trajectory in enumerate(trajectories):
            plt.scatter(*zip(*trajectory), c=colors[inds[i]], label=f'Trajectory {i+1}')
        plt.hlines(y=0, xmin=-lim, xmax=lim, color='black', linestyle='--')
        # Add a vertical line in x=0 for y < 0
        plt.vlines(x=0, ymin=-lim, ymax=0, color='black', linestyle='--')
        # Add two vertical lines in x=0.5 and x=-0.5 for y > 0
        plt.vlines(x=[-0.5, 0.5], ymin=0, ymax=lim, color='black', linestyle='--')
    plt.plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], color='red')
    # Set the axis limits and labels
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory of 2D points')
    # Show the plot
    plt.show()

def plot_zs(zs, case, input=0, lim=2, rows=3, cols=3, s=5):
  figsize = (15,15) if rows == 3 and cols == 3 else (15,7)
  fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
  index = 0
  for row in range(rows):
    for col in range(cols):
      # Get the data for this subplot
      pnts = zs[index].detach().numpy()
      ax[row, col].scatter(pnts[:, 0], pnts[:, 1], color='blue', s=s)
      # Plot the square as a red rectangle
      ax[row, col].plot([-1, -1, 1, 1, -1], [-1, 1, 1, -1, -1], color='red')
      # Set the axis limits and labels
      ax[row, col].set_xlim(-lim, lim)
      ax[row, col].set_ylim(-lim, lim)
      ax[row, col].set_xlabel('x')
      ax[row, col].set_ylabel('y')
      ax[row, col].set_aspect('equal')
      # Set the title for the subplot
      if case == PLOT_SEEDS:
        ax[row, col].set_title(f'Plot seed {input[index]}')
      elif case == PLOT_STAGES:
        ax[row, col].set_title(f'stoped after {input[index]}/{NUM_ITERATIONS} iterations')
      index += 1
  plt.tight_layout()
  plt.show()


def plot_schedulers(T=1000):
    ts = torch.linspace(0,1,T)
    exp = exp_scheduler(ts).detach().numpy()
    pol = pol_scheduler(ts).detach().numpy()
    # Plot the loss as a function of the epochs
    fig, ax = plt.subplots()
    ax.plot(ts, exp, label="exponent scheduler")
    # Plot the second plot on the same axes object
    ax.plot(ts, pol, label="t**2 scheduler")
    ax.legend()
    plt.show()


def plot_conditional_dataset(size=TRAIN_DATASET_SIZE, show=True):
    x0 = get_square_dataset(size)
    c = add_labels(x0)
    print(f"plot_conditional_dataset: {show}")
    visualize_points_con(x0, c, show=show)

"""# Train denoiser """

def get_square_dataset(num_points):
    points = torch.rand(num_points, 2) * 2 - 1
    return points

def exp_scheduler(t):
  return torch.exp(5 * (t - 1))

def cos_scheduler(t):
  return 0.5*(1 - torch.cos(np.pi*t))

def pol_scheduler(t):
  return t**2

def add_labels(x0):
    labels = torch.zeros((x0.shape[0]))
    x = x0[:, 0]
    y = x0[:, 1]
    # assign labels according to the given conditions
    labels[(x < 0) & (y < 0)] = 0
    labels[(x > 0) & (y < 0)] = 1
    labels[(x > 0.5) & (y > 0)] = 2
    labels[(x < -0.5) & (y > 0)] = 3
    labels[(-0.5 < x) & (x < 0.5) & (y > 0)] = 4
    visualize_points_con(x0, labels)
    return labels.int()


def sigma_t(t):
    """
    :param t: a tensor of shape (batch_size, 1) containing the time
    :return: a tensor of shape (batch_size, 1) containing the variance
    """
    if SCHEDULER == EXP_SCHEDULER:
        sig = torch.exp(5 * (t - 1))
    elif SCHEDULER == COS_SCHEDULER:
        sig = 0.5*(1 - torch.cos(np.pi*t))
    elif SCHEDULER == POL_SCHEDULER:
        sig = t**2
    return torch.abs(sig) 


def sigma_dot(t):
    if SCHEDULER == EXP_SCHEDULER:
        sig_dot = 5*sigma_t(t)
    elif SCHEDULER == COS_SCHEDULER:
        sig_dot = 0.5 * np.pi * torch.sin(np.pi * t)
        # sig_dot = torch.sin(t)
    elif SCHEDULER == POL_SCHEDULER:
        sig_dot = 2*(t)
    return torch.abs(sig_dot) 


def train_denoiser(conditional=False):
    x0 = get_square_dataset(TRAIN_DATASET_SIZE)
    D = ConDenoiser() if conditional else Denoiser()
    if conditional:
        c = add_labels(x0)
    # Create an optimizer for the denoiser network parameters
    optimizer = optim.Adam(D.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    # Train the diffusion model for one epoch
    losses = []
    num_epochs = NUM_EPOCHS
    for j in range(num_epochs):
        # number of iterations per epoch
        optimizer.zero_grad()  # reset the gradients
        noise_added = torch.randn_like(x0)  # the noise we add to x0
        t = torch.rand(x0.size(0), 1)  # the t chosen for each point
        xt = x0 + sigma_t(t) * noise_added  # apply Eq. 4 forward diffusion
        # predict the noise in x,y
        noise_predicted = D(xt, t, c) if conditional else D(xt, t)
        loss = criterion(noise_predicted, noise_added)
        losses.append(loss.item())
        loss.backward()  # compute the gradients
        optimizer.step()  # update the denoiser parameters
        if j % 250 == 0:
          print(f"denoiser loss for epoch {j}: {loss.item():.5f}") 
    # plot_denoiser_loss(losses)
    return D, losses

"""# Sampling methods"""

def DDIM_sampeling(D, follow_trajectory=False, num_samples=NUM_SAMPLES, T=T,
                   seed=RANDOM_SEED, vis=False, stop_after=0):
    torch.manual_seed(seed)
    dt = 1/T
    if follow_trajectory:
        z = torch.tensor([[1.9, 1.7]])
        num_samples = 1
        trajectory = []
    else:
        # z = (torch.rand(num_samples, 2, ) - 0.5)*4  # spread equally from -2 < x,y < 2
        z = torch.randn(num_samples, 2, ) # sample from gaussian
    dt_s = np.linspace(1, 0, T + 1)[:-1]
    if vis: visualize_points(z, lim=2)
    for i, t in enumerate(dt_s):
        ts = torch.full((num_samples, 1), t)
        noise_predicted = D(z, ts)
        x0_est = z - sigma_t(ts) * noise_predicted
        score = (x0_est - z)/sigma_t(ts)**2
        dz = sigma_dot(ts) * sigma_t(ts) * score * dt
        z = z + dz
        if follow_trajectory:
            coords = np.squeeze(z.detach().numpy())
            y, x = coords[0], coords[1]
            trajectory.append((x, y))
        if stop_after and i == stop_after and not follow_trajectory:
            break
    if vis: visualize_points(z)
    if follow_trajectory:
        plot_trajectory(dt_s, trajectory, NON_CONDITIONAL)
    return z

def DDIM_sampeling_conditional(D, follow_trajectory=False, num_samples=NUM_SAMPLES, T=T, seed=RANDOM_SEED):
    torch.manual_seed(seed)
    dt = 1/T
    if follow_trajectory:
        z = torch.tensor([[1.9, 1.9], [-1.9, -1.9], [1.9, -1.9], [-1.9, 1.9], [0, 1.9]])
        num_samples = 5
        c = torch.tensor([3,0,2,1,4])
        trajectories = [[] for i in range(num_samples)]
    else:
        # z = (torch.rand(num_samples, 2, ) - 0.5) * 4  # spread equally from -2 < x,y < 2
        z = torch.randn(num_samples, 2, ) # sample from gaussian
        c = torch.randint(low=0, high=5, size=(num_samples,))
    dt_s = np.linspace(1, 0, T + 1)[:-1]
    visualize_points_con(z, c)

    for i, t in enumerate(dt_s):
        ts = torch.full((num_samples, 1), t)
        noise_predicted = D(z, ts, c)
        x0_est = z - sigma_t(ts) * noise_predicted
        score = (x0_est - z)/sigma_t(ts)**2
        dz = sigma_dot(ts) * sigma_t(ts) * score * dt
        z = z + dz
        if follow_trajectory:
            coords = np.squeeze(z.detach().numpy())
            for sample in range(num_samples):
                x, y = coords[sample, 0], coords[sample, 1]
                trajectories[sample].append((x, y))
    visualize_points_con(z, c)
    if follow_trajectory: plot_trajectory(dt_s, trajectories, CONDITIONAL)


def DDPM(D, follow_trajectory=False, num_samples=NUM_SAMPLES, T=T, seed=RANDOM_SEED, lamda=0.001):
    dt = 1/T
    if follow_trajectory:
        z = torch.tensor([[1.9, 1.9], [-1.9, -1.9], [1.9, -1.9], [-1.9, 1.9], [0, 1.9]])
        num_samples = 5
        trajectories = [[] for i in range(num_samples)]
    else:
        # z = (torch.rand(num_samples, 2) - 0.5) * 4  # spread equally from -2 < x,y < 2
        z = torch.randn(num_samples, 2, ) # sample from gaussian
    dt_s = np.linspace(1, 0, T + 1)[:-1]
    visualize_points(z)
    for i, t in enumerate(dt_s):
        ts = torch.full((num_samples, 1), t)
        noise_predicted = D(z, ts)
        x0_est = z - sigma_t(ts) * noise_predicted
        score = (x0_est - z) / sigma_t(ts) ** 2
        g_t = torch.sqrt(2 * sigma_dot(ts) * sigma_t(ts))
        dW = torch.randn(num_samples, 2)
        dz = 0.5 * (1 + lamda**2) * (g_t ** 2) * score * dt + lamda * g_t * dW
        z = z + dz
        if follow_trajectory:
          coords = np.squeeze(z.detach().numpy())
          for sample in range(num_samples):
              x, y = coords[sample, 0], coords[sample, 1]
              trajectories[sample].append((x, y))
    visualize_points(z)
    if follow_trajectory: plot_trajectory(dt_s, trajectories, CONDITIONAL)

def SNR(ts):
  return 1/(sigma_t(ts)**2)

def caculate_probability(D, point, c, T=1000, iterations=1000):
  xs = torch.tile(point, (iterations, )).reshape(iterations, 2)
  ts = torch.rand(T, 1)
  dt = 1/T
  cs = torch.tile(c, (iterations, )).reshape(iterations)
  noise_added = torch.randn_like(xs) 
  xt = xs + sigma_t(ts) * noise_added
  noise_predicted = D(xt, ts, cs)
  x0_est = xt - sigma_t(ts) * noise_predicted
  elbo = (T/2)*((SNR(ts - dt) - SNR(ts))*torch.norm(xs - x0_est)**2).mean()
  print(elbo/1000000)
  pass

"""# Train denoisers"""

SCHEDULER = EXP_SCHEDULER
# SCHEDULER = COS_SCHEDULER
# SCHEDULER = POL_SCHEDULER

# non conditional
denoiser_non_cond, losses_non_c = train_denoiser(conditional=False)
denoiser_cond, losses_cond = train_denoiser(conditional=True)

PLOT_SEEDS = "PLOT_SEEDS"
PLOT_STAGES = "PLOT_STAGES"
NUM_ITERATIONS = 1000

def Q1():
  DDIM_sampeling(denoiser_non_cond, follow_trajectory=True, vis=True)

def Q2():
  plot_denoiser_loss(losses_non_c)
  

def Q3():
  # Present a figure with 9 (3x3 table) different samplings of 1000 points, using 9 different seeds.
  zs = []
  NUM_SEEDS = 9
  for seed in range(NUM_SEEDS):
    z = DDIM_sampeling(denoiser_non_cond, follow_trajectory=False, 
                       T=NUM_ITERATIONS, vis=False, seed=seed)
    zs.append(z)
  lim=1.2
  plot_zs(zs, PLOT_SEEDS, np.arange(NUM_SEEDS), lim)


def Q4():
    zs = []
    NUM_STOPS = 9
    stops = np.array([1, 2, 3, 4, 5, 10, 50, 100, 1000]).astype(np.int)
    for i in range(NUM_STOPS):
      stop_after = stops[i]
      T = stops[i]
      z = DDIM_sampeling(denoiser_non_cond, follow_trajectory=False, T=T, vis=False, 
                        num_samples=NUM_SAMPLES, seed=0, stop_after=T)
      zs.append(z)
    lim=3
    plot_zs(zs, PLOT_STAGES, stops, lim)

def Q5():
  plot_schedulers()

def Q6():
  zs = []
  NUM_RUNS=10
  for i in range(10):
      z = DDIM_sampeling(denoiser_non_cond, follow_trajectory=False, T=100, vis=False, 
                        num_samples=NUM_SAMPLES, seed=0,)
      zs.append(z)
  plot_zs(zs, "", np.arange(NUM_SAMPLES), lim=1.2, rows=2, cols=5)
  # use another method for differnt outputs in ebery iteration
  DDPM(denoiser_non_cond, follow_trajectory=True, num_samples=NUM_SAMPLES, T=100, seed=0, lamda=0.01)
  DDPM(denoiser_non_cond, follow_trajectory=False, num_samples=NUM_SAMPLES, T=100, seed=0, lamda=0.01)

"""## Answer questions

# Conditional questions
"""

def cQ1():
  plot_conditional_dataset()

def cQ3():
  DDIM_sampeling_conditional(denoiser_cond,T=2000, follow_trajectory=True)

def cQ4():
  DDIM_sampeling_conditional(denoiser_cond,T=5000, follow_trajectory=False)

def cQ6():
  points = torch.tensor([[-0.5, -0.5], [-0.5, -0.5], [-0.5, 1.5], [0.5, 0.75], [-0.9, -0.9]])   
  C = torch.tensor([0, 1, 0, 2, 0])
  colors = ["red", "green", "blue", "black", "purple"]
  plot_conditional_dataset(1000, False)
  for i in range(len(points)):
    pnt = points[i, :]
    clss = C[i]
    P = caculate_probability(denoiser_cond, pnt, clss)
    plt.scatter(pnt[1], pnt[0], marker='o',s=200, c=colors[C[i]])
  pnt = points[0, :]
  c = C[0]
  plt.scatter(pnt[1], pnt[0], marker='o',s=100, c=c)
  plt.show()