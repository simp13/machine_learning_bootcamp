# import needed library
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)
    
    def forward(self,item):
        x = self.fc1(item)
        x = self.fc2(x)
        return x

# initailize model
# model = NN(784,10)
# x = torch.rand(64,784)
# print(model(x).shape)

# device gpu or cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
epochs = 1

# Load Data
train_dataset = datasets.MNIST(root="dataset/",train=True,transform=transforms.ToTensor(),download=True)
training_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

valid_dataset = datasets.MNIST(root="dataset/",train=False,transform=transforms.ToTensor(),download=True)
valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=True)

# initialize netowrk
model = NN(784,10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


training_losses = []
validation_losses = []

# training network
for epoch in range(epochs):
    # training dataset
    total_train_loss = 0
    total_valid_loss = 0
    for batchindex,(data,target) in enumerate(tqdm(training_loader)):
        data = data.to(device)
        target = target.to(device)
        # get data to correct shape
        data = data.reshape(data.shape[0],-1)
        # forward
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        # backward
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    training_losses.append(total_train_loss / len(training_loader))


    # validation dataset
    for batchindex,(data,target) in enumerate(tqdm(valid_loader)):
        data = data.to(device)
        target = target.to(device)
        # get data to correct shape
        data = data.reshape(data.shape[0],-1)
        model.eval()
        with torch.no_grad():
            output = model(data)
            loss = criterion(output,target)
            total_valid_loss += loss.item()
    validation_losses.append(total_valid_loss / len(valid_loader))



def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0],-1)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy of {float(num_correct)/float(num_samples) * 100:.2f}")

    model.train()
    return float(num_correct)/float(num_samples)

print("[INFO] Checking Accuracy")
check_accuracy(training_loader,model)
check_accuracy(valid_loader,model)

print("Total loss for training",training_losses)
print("Total loss for validation",validation_losses)
# plot losses
ax,fig = plt.subplots()
plt.plot(training_losses)
plt.plot(validation_losses)
plt.show()