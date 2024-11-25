# -*- coding: utf-8 -*-
"""example_Anomaly.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11F9cbuS0GWldcNtJjC2xmFgMqnwJVlBX
"""

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from sklearn.metrics import roc_curve, roc_auc_score
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset,TensorDataset
import torchvision
import torch
!pip install faiss-cpu
import faiss
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import resnet18

batch_size=256
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
device = torch.device('cuda')

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
    path_of_encoder = "enc_40_mu_25.pth"
    encoder_augment = torch.load(path_of_encoder)
    encoder_augment.eval() # Set the model to evaluation mode

    reps_augment_test, labels_augment_test = get_representations(encoder_augment, mnist_cifar_loader)
    reps_augment_train, labels_augment_train = get_representations(encoder_augment, trainloader)

    fpr_a, tpr_a, auc_a, knn_density_a = find_knn_density(reps_augment_test, reps_augment_train)

    plt.plot(fpr_a, tpr_a, color='orange', label='ROC augmented')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    # plot the AUC score
    plt.text(0.5, 0.2, f'AUC augmented = {auc_a:.2f}', fontsize=12)
    plt.show()

    # get the 7 most anomalous samples
    anoumalous_inds_a = np.argsort(knn_density_a)[-7:]
    anomalous_a = []
    for ind in anoumalous_inds_a:
      anomalous_a.append(mnist_cifar[ind])

    fig, axs = plt.subplots(nrows=len(anomalous_a), ncols=1, figsize=(10, 10))
    for i in range(len(anomalous_a)):
        axs[i].imshow(anomalous_a[i][0].numpy().transpose([1,2,0]))
    plt.show()

run_anomaly_detection()