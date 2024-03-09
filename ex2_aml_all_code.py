# -*- coding: utf-8 -*-
"""EX2 AML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f-o1ozH_kNeKXkxRkjKi5I5wv7sQ8zjV
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
# save_path_dir_model = "/content/drive/MyDrive/AML course/model/"
# save_path_dir = "/content/drive/MyDrive/AML course/graphs/"

save_path_dir = r"/content/drive/MyDrive/DL course/EX2/graphs/"
save_path_dir_model = r"/content/drive/MyDrive/DL course/EX2/trained models/"

# Commented out IPython magic to ensure Python compatibility.
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
!pip install hnswlib
import hnswlib

!pip install faiss-cpu
import faiss
# %matplotlib inline

"""# Augmentations"""

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
device = torch.device('cuda')
AUGMENTATIONS = "AUGMENTATIONS"
NEIGHBORS = "NEIGHBORS"

"""# Load datasets"""

batch_size = 256
transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

"""# Encoder and Projection models"""

class Encoder(nn.Module):
    def __init__(self, D=128, device='cuda'):
        super(Encoder, self).__init__()
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(512, 512)
        self.fc = nn.Sequential(nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, D))

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def encode(self, x):
        return self.forward(x)


class Projector(nn.Module):
    def __init__(self, D=128, proj_dim=512):
        super(Projector, self).__init__()
        self.model = nn.Sequential(nn.Linear(D, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim),
                                   nn.BatchNorm1d(proj_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(proj_dim, proj_dim)
                                   )

    def forward(self, x):
        return self.model(x)

def inv_L(input, target):
  return F.mse_loss(input, target)

def var_per_z(Z, gamma=1, eps=10e-4):
  var = torch.var(Z, dim=0)
  std = torch.sqrt(var + eps)
  loss = torch.mean(F.relu(gamma - std))  # max(g-std, 0)
  return loss

def get_var_loss(input, target):
  loss_input = var_per_z(input)
  loss_target = var_per_z(target)
  return loss_input + loss_target


def cov_per_z(Z):
  Z = Z - Z.mean(dim=0)
  bs, emb = Z.shape[0], Z.shape[1]
  cov_z = ((Z.T @ Z) / (bs - 1)).square()  # DxD
  loss_c2 = (cov_z.sum() - cov_z.diagonal().sum()) / emb
  return loss_c2

def get_cov_loss(input, target):
  loss_input = cov_per_z(input)
  loss_target = cov_per_z(target)
  return loss_input + loss_target

def VICReg_loss(input, target):
  return inv_L(input, target), get_var_loss(input, target), get_cov_loss(input, target)

def get_B1_B2_different_views(batch):
    B1 = torch.stack([train_transform(b) for b in batch]).to(device)
    B2 = torch.stack([train_transform(b) for b in batch]).to(device)
    return B1, B2

"""Criterion noam

# plot
"""

# Plot the train and test losses
def show_and_save_graph(X, label, title,xlabel='batches'):
  plt.plot(X, label=label)
  plt.legend()
  plt.xlabel(xlabel)
  plt.ylabel('Loss')
  plt.title(title)
  plt.savefig(f'{save_path_dir}{title}.png')
  plt.show()


def plot_train_vs_test(train, test,title="", label_train="train", label_test="test"):
    test_losses = np.array(test)
    train_losses = np.array(train)
    test_losses_interpolated = np.interp(np.linspace(0, 1, len(train_losses)), np.linspace(0, 1, len(test_losses)), test_losses)
    print(test_losses_interpolated.shape, train_losses.shape)
    num_batches = 196
    plt.plot(test_losses_interpolated, label=f"{label_test}", linewidth=2)
    plt.plot(train_losses, label=f"{label_train}", linewidth=0.2)
    plt.legend()
    plt.xlabel("train")
    plt.ylabel('Loss')
    plt.title(f"{title}")
    plt.savefig(f'{save_path_dir}{title}.png')
    plt.show()

"""Train VICReg

# Q1: Training
Train VICReg on the CIFAR10 dataset. Plot the values of each of the 3 objectives (in separate
figures) as a function of the training batches. In your figures also include the loss terms values on the test set,
computed once every epoch. Note: On Google Colab’s free tier GPU, this training should take about 30 minutes.
"""

def train_encoder(num_epochs=30, var_loss=True, show_graphs=True):
    enc = Encoder().to(device)
    proj = Projector().to(device)
    lr = 3*10e-4
    betas = (0.9, 0.999)
    weight_decay = 10e-6
    params = [{'params': enc.parameters()}, {'params': proj.parameters()}]
    optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    criterion = VICReg_loss
    inv_losses_train, var_losses_train, cov_losses_train, train_losses = [],[],[],[]
    inv_losses_test, var_losses_test, cov_losses_test, test_losses = [],[],[],[]
    num_epochs = num_epochs
    L = 25
    M = 25 if var_loss else 0
    V = 1
    for epoch in range(num_epochs) :  # loop over the dataset multiple times
        print(f"epoch {epoch+1}")
        num_batches_train = len(trainloader)
        num_batches_test = len(testloader)

        # test on test set
        with torch.no_grad():
          inv_loss_test, var_loss_test, cov_loss_test, test_loss = 0,0,0,0
          for batch_num, batch in tqdm(enumerate(testloader, 0),total=len(testloader)):
              B, labels = batch
              B1, B2 = get_B1_B2_different_views(B)
              Z1, Z2 = proj(enc(B1)), proj(enc(B2))
              inv_loss, var_loss, cov_loss = criterion(Z1, Z2)
              loss = L*inv_loss + M*var_loss + V*cov_loss

              inv_loss_test += inv_loss.item()
              var_loss_test += var_loss.item()
              cov_loss_test += cov_loss.item()
              test_loss += loss.item()
          inv_loss_test /= num_batches_test
          var_loss_test /= num_batches_test
          cov_loss_test /= num_batches_test
          test_loss /= num_batches_test

          inv_losses_test.append(inv_loss_test)
          var_losses_test.append(var_loss_test)
          cov_losses_test.append(cov_loss_test)
          test_losses.append(test_loss)

        for batch_num, batch in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
            B, labels = batch
            B1, B2 = get_B1_B2_different_views(B)
            Z1, Z2 = proj(enc(B1)), proj(enc(B2))
            inv_loss, var_loss, cov_loss = criterion(Z1, Z2)
            loss = L*inv_loss + M*var_loss + V*cov_loss
            # loss, inv_loss, var_loss, cov_loss = criterion(Z1, Z2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save statistics
            inv_losses_train.append(inv_loss.item())
            var_losses_train.append(var_loss.item())
            cov_losses_train.append(cov_loss.item())
            train_losses.append(loss.item())

    torch.save(enc, f'{save_path_dir_model}enc_{num_epochs}_mu_{M}.pth')
    torch.save(proj, f'{save_path_dir_model}proj_{num_epochs}_mu_{M}.pth')
    if show_graphs:
      # show and save training graphs
      plot_train_vs_test(train_losses, test_losses, title=f"Train loss mu {M}")
      plot_train_vs_test(inv_losses_train, inv_losses_test, title=f"inv loss mu {M}")
      plot_train_vs_test(var_losses_train, var_losses_test, title=f"var loss mu {M}")
      plot_train_vs_test(cov_losses_train, cov_losses_test, title=f"cov loss mu {M}")

# train_encoder(num_epochs=10, var_loss=False, show_graphs=True)
# train_encoder(num_epochs=40, var_loss=True, show_graphs=True)

"""# Graphs

# Q2: PCA vs T-SNE Visualizations
Compute the representations of each test image using your trained encoder.
Map (using the sklearn library) the representation to a 2D space using:

 (i) PCA

  (ii) T-SNE [4].

  Plot the T-SNE and the PCA 2D representations, colored by their classes. Look at both visualizations (PCA vs. T-SNE), which one
seems more effective for visualizations to you? Look at the T-SNE visualization. Did VICReg managed to capture
the class information accurately? Which classes seem entangled to you? Explain your answer in detail.
"""

def load_encoder(var_loss=True, neighbors=False):
  # load the trained model
  if var_loss and not neighbors:
    path_of_encoder = "/content/drive/MyDrive/DL course/EX2/trained models/enc_40_mu_25.pth"
  elif not var_loss and not neighbors:
    path_of_encoder = r"/content/drive/MyDrive/DL course/EX2/trained models/enc_10_mu_0.pth"
  elif var_loss and neighbors:
    path_of_encoder = r"/content/drive/MyDrive/DL course/EX2/trained models/neighbors_enc_5.pth"
  print(path_of_encoder)
  encoder = torch.load(path_of_encoder)
  encoder.eval() # Set the model to evaluation mode
  for param in encoder.parameters():
    encoder.requires_grad = False
  return encoder

def get_representations(encoder, dataloader):
  representations = []
  class_labels = []
  for batch_idx, (batch, labels) in enumerate(dataloader):
      encoded_data = encoder.encode(batch.to(device))
      for i in range(len(encoded_data)):
          representations.append(encoded_data[i].tolist())
          class_labels.append(labels[i].tolist())
  representations = torch.tensor(representations)
  class_labels = torch.tensor(class_labels)
  print(f"representations is of size {representations.size()} and the labels are of size {class_labels.size()}")
  return representations, class_labels

def find_pca_TSNE(title=""):
  encoder = load_encoder(var_loss=False)
  representations, class_labels = get_representations(encoder, testloader)
  # Compute T-SNE 2D representation for each element of A
  tsne = TSNE(n_components=2)
  representations_tsne = tsne.fit_transform(representations)
  M=25
  # Compute PCA 2D representation for each element of A
  pca = PCA(n_components=2)
  representations_pca = pca.fit_transform(representations)
  # Display the T-SNE and PCA 2D representations using different colors for each label
  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  save_path = "/content/drive/MyDrive/DL course/EX2/graphs/"
  fig, ax = plt.subplots()
  scatter = ax.scatter(representations_tsne[:,0], representations_tsne[:,1], c=class_labels, s=2)
  legend1 = ax.legend(*scatter.legend_elements(), title="Classes", loc='upper right')
  for i, txt in enumerate(classes):
      legend1.get_texts()[i].set_text(txt)
  ax.add_artist(legend1)
  ax.set_title(f'T-SNE 2D representation {title}')
  min_val = -90
  max_val = 170
  plt.xlim(min_val, max_val)
  plt.ylim(min_val, max_val)
  # plt.axis('equal')

  plt.subplots_adjust(right=0.7)
  plt.savefig(f'{save_path}T-SNE.png')
  plt.show()


  fig, ax = plt.subplots()
  scatter = ax.scatter(representations_pca[:,0], representations_pca[:,1], c=class_labels, s=2)
  legend2 = ax.legend(*scatter.legend_elements(), title="Classes", loc='upper right')
  for i, txt in enumerate(classes):
      legend2.get_texts()[i].set_text(txt)
  ax.add_artist(legend2)
  # min_val = -80
  # max_val = 130
  # plt.xlim(min_val, max_val)
  # plt.xlim(min_val, max_val)
  ax.set_title(f'PCA 2D representation {title}')
  plt.subplots_adjust(right=0.7)
  # plt.axis('equal')
  plt.savefig(f'{save_path}PCA.png')
  plt.show()

# find_pca_TSNE(title="VICReg no variance")

"""# Q3: Linear Probing

Perform a linear probing (single FC layer) to the encoder’s representation.

Train this classifier on the representations of the CIFAR10 train set. Remember to freeze the encoder, i.e. do not update it.

Compute the probing’s accuracy on the test set.

What is the accuracy you reach with your classifier?
Note:

classifier accuracy should be at least 60% on the test set
"""

class LinearProbing(nn.Module):
    def __init__(self, encoder, D=128, C=10):
        # D is the encoded latent space and C is the number of classes
        super(LinearProbing, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(D, C)
    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

import torch.optim as optim

def test(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            labels = labels.to(device)
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

def train_and_test_LP():
  encoder = load_encoder(neighbors=True)
  encoder.eval()
  for param in encoder.parameters():
    # Set the requires_grad attribute to False
    param.requires_grad = False
  LP_model = LinearProbing(encoder).to(device)
  print(sum(p.numel() for p in LP_model.parameters() if p.requires_grad))
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(LP_model.parameters())


  # train the linear probing model
  losses_LP = []
  num_epochs = 20
  for epoch in range(num_epochs) :  # loop over the dataset multiple times
      num_batches_train = len(trainloader)
      for batch_num, batch in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
          B, labels = batch
          pred_labels = LP_model(B.to(device))
          loss = criterion(pred_labels, labels.to(device))
          losses_LP.append(loss.item())
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
  test(LP_model, testloader)
  torch.save(LP_model, f'{save_path_dir_model}linear_probing_{num_epochs}.pth')
  show_and_save_graph(losses_LP, "Linear Probing Loss", "Linear Probing Loss")
train_and_test_LP()

"""# Q6: Ablation
 Now, we Generated Neighbors Now, we would like to ablate VICReg by only removing the generated
neighbors, using neighbors from the data itself: First, compute the representations of your original VICReg, on all
of the training set. In each step of training and for each image in the batch, use these representations to find the top
3 nearest neighbors, and randomly select 1 of them. Use the original image and this neighbor of it as your 2 views

for the VICReg algorithm. 2 Practical Tips:

(i) We find that training this algorithm for only a single epoch is more
beneficial.

(ii) We recommend you to compute the neighboring indices of each image in advance, and delete the
original VICReg model from your (GPU) memory. This will save both run time and GPU space.
Compute the linear probing accuracy, and report it. Is the accuracy different from the original linear probing from
Q3?


• If no, explain what added value do you think the generated neighbors adds to the algorithm.


• If yes, explain why do you think this change had no effect (what compensates the things that are missing?).
"""

from torch.utils.data import DataLoader, TensorDataset
import random
import hnswlib
def get_Q6_dataloader():
  encoder = load_encoder()
  representations, class_labels = get_representations(encoder, trainloader)
  dim = representations.shape[1]
  num_elements = representations.shape[0]
  print(num_elements)
  index = faiss.IndexFlatL2(dim)
  index.add(representations)
  distance, neighbors_3 = index.search(representations, 3)
  q6_train_data = []
  for i, im in enumerate(trainset):
      if i == num_elements: break
      train_neighbor = np.array(trainset[neighbors_3[i, np.random.randint(3)]][0])
      image = np.array(im[0])
      q6_train_data.append([image, train_neighbor])

  q6_data_set = TensorDataset(torch.as_tensor([c[0] for c in q6_train_data]), torch.as_tensor([c[1] for c in q6_train_data]))
  q6_data_loader = DataLoader(q6_data_set, batch_size=batch_size, shuffle=True)
  return q6_data_loader

def Q6_train_neighbors_based_encoder():
    Q6_dataloader = get_Q6_dataloader()
    enc = Encoder().to(device)
    proj = Projector().to(device)
    lr = 3*10e-4
    betas = (0.9, 0.999)
    weight_decay = 10e-6
    params = [{'params': enc.parameters()}, {'params': proj.parameters()}]
    optimizer = torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    criterion = VICReg_loss
    inv_losses_train, var_losses_train, cov_losses_train, train_losses = [],[],[],[]
    num_epochs = 1
    L, M, V = 25, 25, 1
    for epoch in range(num_epochs) :  # loop over the dataset multiple times
        print(f"epoch {epoch+1}")
        for i, batch in tqdm(enumerate(Q6_dataloader), total=len(Q6_dataloader)):
            B1 = batch[0].to(device)
            B2 = batch[1].to(device)
            Z1, Z2 = proj(enc(B1)), proj(enc(B2))
            inv_loss, var_loss, cov_loss = criterion(Z1, Z2)
            loss = L*inv_loss + M*var_loss + V*cov_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

    show_and_save_graph(train_losses, "Train loss", f"Train loss Q6")
    torch.save(enc, f'{save_path_dir_model}neighbors_enc_{num_epochs}.pth')
Q6_train_neighbors_based_encoder()

"""# Q8 Retrieval Evaluation

Now that we slowly "pealed" VICReg back to the laplacian eigenmaps algorithm,
we wish to evaluate it qualitatively. For the methods Q1 (VICReg) and Q6 (No Generated Neighbors) perform a
qualitative retrieval evaluation. That means:
• Select 10 random images from the training set, one from each class.


• For each selected image, use the representations of each of the evaluated   methods, to find its 5 nearest neighbors in the dataset.


• Plot the images together with their neighbors.


• Using the same images, perform the same visualization for the 5 most distant images in the dataset.

"""

# start with the
def disply_nearest_neighbors(encoder, reps):
  class_images = {}
    # Loop through the dataset and add one image from each class to the dictionary
  for i in range(len(testset)):
      image, label = testset[i]
      if label not in class_images:
          class_images[label] = image
      if len(class_images) == 10: break
  all_neighbors_images = []
  all_far_neghbors_images = []

  for label, image in class_images.items():
      encoded = encoder(image.unsqueeze(0).to(device))[0].unsqueeze(0).to(device)
      distances = torch.cdist(encoded, reps.to(device))
      # Find the indices of the 5 nearest neighbors
      _, indices = distances.topk(k=5, largest=False)
      inds = indices[0].cpu().detach().numpy()
      # Get the 5 nearest neighbors
      image = image.numpy().transpose([1,2,0])
      neighbor_images = [image]
      for i in inds:
          neighbor_img = testset[i][0].numpy().transpose([1,2,0])
          neighbor_images.append(neighbor_img)
      all_neighbors_images.append(neighbor_images)

      # far neighbors
      _, far_indices = distances.topk(k=5, largest=True)
      far_inds = far_indices[0].cpu().detach().numpy()
      # Get the 5 nearest neighbors
      far_neighbor_images = [image]
      for i in far_inds:
          far_neighbor_img = testset[i][0].numpy().transpose([1,2,0])
          far_neighbor_images.append(far_neighbor_img)
      all_far_neghbors_images.append(far_neighbor_images)

  # plot the close neighbors
  num_classes = len(classes)
  num_images_per_row = 6
  fig, axs = plt.subplots(nrows=num_classes, ncols=num_images_per_row, figsize=(15, 15))
  print(len(all_neighbors_images), len(all_neighbors_images[0]))
  # Loop over each row and column and display the image
  for i in range(num_classes):
      for j in range(num_images_per_row):
          axs[i,j].imshow(all_neighbors_images[i][j])
          axs[i,j].axis('off')
  plt.show()

  # plot the far neighbors
  num_classes = len(classes)
  num_images_per_row = 6
  fig, axs = plt.subplots(nrows=num_classes, ncols=num_images_per_row, figsize=(15, 15))
  print(len(all_neighbors_images), len(all_neighbors_images[0]))
  # Loop over each row and column and display the image
  for i in range(num_classes):
      for j in range(num_images_per_row):
          axs[i,j].imshow(all_far_neghbors_images[i][j])
          axs[i,j].axis('off')
  plt.show()


def find_the_neighbors():
    # encoder with var loss
    encoder_with_var = load_encoder(var_loss=True).to(device)
    encoder_no_var = load_encoder(var_loss=False).to(device)
    encoder_neighbors = load_encoder(neighbors=True).to(device)

    reps_with_var, labels_var = get_representations(encoder_with_var, testloader)
    # reps_no_var, labels_no_var = get_representations(encoder_no_var, testloader)
    reps_neighbors, labels_neighbors = get_representations(encoder_neighbors, testloader)

    disply_nearest_neighbors(encoder_with_var, reps_with_var)
    # disply_nearest_neighbors(encoder_no_var, reps_no_var)
    disply_nearest_neighbors(encoder_neighbors, reps_neighbors)


find_the_neighbors()

"""# Downstream Applications

# Anomaly Detection
"""

from sklearn.metrics import roc_curve, roc_auc_score
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset,TensorDataset


def find_knn_density(test_reps, train_reps):
  dim = test_reps.shape[-1]
  index = faiss.IndexFlatL2(dim)
  index.add(train_reps) # serach in the training set representations
  distance, neighbors_2 = index.search(test_reps, 2)
  knn_density = np.mean(distance, axis=-1)
  labels = np.zeros(knn_density.shape[0],)
  labels[:10000] = 1
  fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=knn_density)
  auc = roc_auc_score(y_true=labels, y_score=knn_density)
  return fpr, tpr, auc, knn_density

def run_anomaly_detection():
    mnist_testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transforms.Compose([transforms.Resize(32), transforms.ToTensor(),
                                                                                       transforms.Lambda(lambda x: x.repeat(3, 1, 1))]))

    mnist_cifar = mnist_testset + testset
    mnist_cifar_loader = torch.utils.data.DataLoader(mnist_cifar, batch_size=batch_size,num_workers=2)

    encoder_augment = load_encoder(var_loss=True).to(device)
    # encoder_neighbors = load_encoder(neighbors=True).to(device)
    encoder_neighbors = torch.load("/content/drive/MyDrive/DL course/EX2/trained models/neighbors_enc_5.pth")
    encoder_neighbors.eval()
    reps_augment_test, labels_augment_test = get_representations(encoder_augment, mnist_cifar_loader)
    reps_augment_train, labels_augment_train = get_representations(encoder_augment, trainloader)

    reps_neighbors_test, labels_neighbors_test = get_representations(encoder_neighbors, mnist_cifar_loader)
    reps_neighbors_train, labels_neighbors_train = get_representations(encoder_neighbors, trainloader)

    fpr_a, tpr_a, auc_a, knn_density_a = find_knn_density(reps_augment_test, reps_augment_train)
    fpr_n, tpr_n, auc_n, knn_density_n = find_knn_density(reps_neighbors_test, reps_neighbors_train)

    plt.plot(fpr_a, tpr_a, color='orange', label='ROC augmented')
    plt.plot(fpr_n, tpr_n, color='blue', label='ROC neighbors')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    # plot the AUC score
    plt.text(0.5, 0.2, f'AUC augmented = {auc_a:.2f}', fontsize=12)
    plt.text(0.5, 0.3, f'AUC neighbors = {auc_n:.2f}', fontsize=12)
    plt.show()

    # get the 7 most anomalous samples
    anoumalous_inds_a = np.argsort(knn_density_a)[-7:]
    anoumalous_inds_n = np.argsort(knn_density_n)[-7:]
    anomalous_a = []
    anomalous_n = []
    for ind in anoumalous_inds_a:
      anomalous_a.append(mnist_cifar[ind])
    for ind in anoumalous_inds_n:
      anomalous_n.append(mnist_cifar[ind])


    fig, axs = plt.subplots(nrows=len(anomalous_n), ncols=2, figsize=(10, 10))
    for i in range(len(anomalous_a)):
        axs[i][0].imshow(anomalous_a[i][0].numpy().transpose([1,2,0]))
        axs[i][1].imshow(anomalous_n[i][0].numpy().transpose([1,2,0]))

    plt.show()

run_anomaly_detection()

"""# (Bonus Section) Clustering

"""

from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestCentroid

def display_tsne(tsne_reps, labels, centroids, title=""):


    classes = ('plane', 'car', 'bird', 'cat',
              'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    save_path = "/content/drive/MyDrive/DL course/EX2/graphs/"
    fig, ax = plt.subplots()
    # show all samples
    scatter1 = ax.scatter(tsne_reps[:,0], tsne_reps[:,1], c=labels, s=1)
    # show centroinds
    scatter2 = ax.scatter(centroids[:,0], centroids[:,1], s=50, edgecolors='black')

    legend1 = ax.legend(*scatter1.legend_elements(), title="Classes", loc='upper right')
    for i, txt in enumerate(classes):
        legend1.get_texts()[i].set_text(txt)
    ax.add_artist(legend1)
    ax.set_title(f'T-SNE 2D {title}')
    min_val = -90
    max_val = 170
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    # plt.axis('equal')

    plt.subplots_adjust(right=0.7)
    plt.savefig(f'{save_path}T-SNE.png')
    plt.show()


def get_tsne_clusterisng(reps, class_labels):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10, random_state=0).fit(reps)
    # Get the cluster labels and centroids
    clustering_labels = kmeans.labels_
    kmeans_centroids = kmeans.cluster_centers_
    sil_score_clust = silhouette_score(reps, clustering_labels)
    sil_score_class = silhouette_score(reps, class_labels)
    print(f"sil score for the clustring labels = {sil_score_clust}")
    print(f"sil score for the real class labels = {sil_score_class}")
    reps_and_cen = torch.cat((reps, torch.tensor(kmeans_centroids)))
    class_labels = torch.cat((torch.tensor(class_labels), torch.arange(10)))
    clustering_labels = torch.cat((torch.tensor(clustering_labels), torch.arange(10)))
    kmeans = KMeans(n_clusters=10, random_state=0).fit(reps_and_cen)
    tsne = TSNE(n_components=2)
    representations_tsne = tsne.fit_transform(reps_and_cen)
    kmeans_centroids = representations_tsne[-10:, :]
    # Create and fit a nearest centroid classifier
    clf = NearestCentroid()
    clf.fit(representations_tsne, class_labels)
    # Get the centroids of each class
    real_centroids = clf.centroids_ # array of shape (10, 2)

    display_tsne(representations_tsne, class_labels, real_centroids, "class labels")
    display_tsne(representations_tsne, clustering_labels, kmeans_centroids,"clustering labels")

encoder_with_var = load_encoder(var_loss=True).to(device)
encoder_neighbors = load_encoder(neighbors=True).to(device)

reps_with_var, labels_var = get_representations(encoder_with_var, testloader)
reps_neighbors, labels_neighbors = get_representations(encoder_neighbors, testloader)

# get_tsne_clusterisng(reps_with_var, labels_var)
get_tsne_clusterisng(reps_neighbors, labels_neighbors)