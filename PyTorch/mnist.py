#importing modules
from os import access
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

#Creating a fully connected network
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,64)
        self.fc2 = nn.Linear(64,num_classes)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.01
batch_size = 64
num_epochs = 10

#Load Data
train_dataset = datasets.MNIST(root = '.dataset/',train = True,transform = transforms.ToTensor(),download = True)
train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
test_dataset = datasets.MNIST(root = '.dataset/',train = False,transform = transforms.ToTensor(),download = True)
test_loader = DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = True)

#initialize network
model = NN(input_size = input_size,num_classes = num_classes).to(device)

#loss and optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

#Train Network
for epoch in range(num_epochs):
    for data,targets in train_loader:
        data = data.to(device = device)
        targets = targets.to(device = device)

        data = data.reshape(data.shape[0],-1)
        scores = model(data)
        loss = criterion(scores,targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

def check_accuracy(loader,model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _,pred = scores.max(1)
            num_correct+=(pred==y).sum()
            num_samples+=pred.size(0)
        
        print(f"got {float(num_correct)/float(num_samples)*100:.2f} accuracy")

    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)

'''
Output:
Checking accuracy on training data
got 98.22 accuracy
Checking accuracy on test data
got 96.40 accuracy
'''