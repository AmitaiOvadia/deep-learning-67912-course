import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

NAM_LABELS = 5
NON_CONDITIONAL = "NON_CONDITIONAL"
CONDITIONAL = "CONDITIONAL"

def get_square_dataset(num_points):
    # sample uniformly num_points points -1 < x,y < 1
    points = torch.rand(num_points, 2) * 2 - 1
    dataset = TensorDataset(points)
    dataloader = DataLoader(dataset, batch_size=num_points, shuffle=True)
    return points


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


def sigma_t(t):
    """
    :param t: a tensor of shape (batch_size, 1) containing the time
    :return: a tensor of shape (batch_size, 1) containing the variance
    """
    return torch.exp(5 * (t - 1))


def sigma_dot(t):
    return 5*sigma_t(t)


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
    # plot_points(x0, labels)
    return labels.int()


def visualize_points_con(x0, labels, lim=2):
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
    plt.show()


def train_denoiser(conditional=False):
    x0 = get_square_dataset(3000)
    D = ConDenoiser() if conditional else Denoiser()
    if conditional:
        c = add_labels(x0)
    # Create an optimizer for the denoiser network parameters
    optimizer = optim.Adam(D.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    # Train the diffusion model for one epoch
    losses = []
    num_epochs = 1000
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
    plot_denoiser_loss(losses)
    return D


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


def DDIM_sampeling(D, follow_trajectory=False, num_samples=1000, T=100, seed=0):
    torch.manual_seed(seed)
    dt = 1/T
    if follow_trajectory:
        z = torch.tensor([[1.9, 1.7]])
        num_samples = 1
        trajectory = []
    else:
        z = (torch.rand(num_samples, 2, ) - 0.5)*4  # spread equally from -2 < x,y < 2
    dt_s = np.linspace(1, 0, T + 1)
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

        if i % 10 == 0 and not follow_trajectory:
            visualize_points(z, lim=2)
    if follow_trajectory:
        plot_trajectory(dt_s, trajectory, NON_CONDITIONAL)


def DDIM_sampeling_conditional(D, follow_trajectory=False, num_samples=1000, T=100, seed=0):
    torch.manual_seed(seed)
    dt = 1/T
    if follow_trajectory:
        z = torch.tensor([[1.9, 1.9], [-1.9, -1.9], [1.9, -1.9], [-1.9, 1.9], [0, 1.9]])
        num_samples = 5
        trajectories = [[] for i in range(num_samples)]
    else:
        z = (torch.rand(num_samples, 2, ) - 0.5) * 4  # spread equally from -2 < x,y < 2
    dt_s = np.linspace(1, 0, T + 1)
    for i, t in enumerate(dt_s):
        c = add_labels(z)
        # if i % 5 == 0: visualize_points_con(z, c)
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
    plot_trajectory(dt_s, trajectories, CONDITIONAL)


def plot_trajectory(dt_s, trajectories, type):
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
        colors = [cmap(i / (n - 1)) for i in range(n)]
        for i, trajectory in enumerate(trajectories):
            plt.scatter(*zip(*trajectory), c=[colors[i]], label=f'Trajectory {i+1}')
    lim = 2
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


if __name__ == "__main__":
    denoiser = train_denoiser(conditional=False)
    DDIM_sampeling(denoiser, follow_trajectory=True)