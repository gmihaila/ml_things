import os
import time
os.environ["CUDA_VISIBLE_DEVICES"]="0"  #<------ CHECK YOUR MLP EMAIL
import torch
import multiprocessing
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.backends import cudnn

cudnn.benchmark = True

## if want to run on multi core
n_cores = 0 # multiprocessing.cpu_count()

# transform the raw dataset into tensors and normalize them in a fixed range
_tasks = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

## Load MNIST Dataset and apply transformations
mnist = MNIST("data", download=True, train=True, transform=_tasks)

## create training and validation split 
split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

## create sampler objects using SubsetRandomSampler
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

## create iterator objects for train and valid datasets
trainloader = DataLoader(mnist, batch_size=256, sampler=tr_sampler, num_workers=n_cores)
validloader = DataLoader(mnist, batch_size=256, sampler=val_sampler, num_workers=n_cores)

## GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Build class of model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        return x

model = Model()

## Multi GPU
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs")
  model = nn.DataParallel(model, device_ids=[0])

model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)

for epoch in range(1,11):

  train_loss, valid_loss = [], []
  model.train() # activates training mod

  ## Training on 1 epoch
  for data, target in trainloader:

    data = torch.flatten(data.to(device), start_dim=1)

    optimizer.zero_grad() #clears gradients of all optimized classes

    ## forward pass
    output = model(data.to(device))

    ## loss calc
    loss = loss_function(output.to(device), target.to(device))

    ## backeard propagation
    loss.backward()

    ## weight optimization
    optimizer.step() #performs a single optimization step
    train_loss.append(loss.item())

  ### Evaluation on 1 epoch
  for data, target in validloader:

    data = torch.flatten(data, start_dim=1)

    output = model(data.to(device))
    loss = loss_function(output.to(device), target.to(device))
    valid_loss.append(loss.item())
  
  print("Epoch:", epoch, "Training Loss: ", np.mean(train_loss), "Valid Loss: ", np.mean(valid_loss))
