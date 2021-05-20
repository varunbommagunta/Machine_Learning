# importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd

# setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
epochs = 10
batch_size = 4  # Set to higher values if more memory is available
learning_rate = 0.01

# data transformations and dataloading
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_dataset = datasets.CIFAR10(
    root=getcwd(), train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root=getcwd(), train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def imshow(img):
    img = (img / 2) + 0.5  # revert
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  #  CHW to HWC
    plt.show()


data_iter = iter(train_loader)
images, targets = data_iter.next()

# imshow(torchvision.utils.make_grid(images))

# Define the network
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(
            16 * 5 * 5, 120
        )  # (32,32,3)->(28,28,6)->(14,14,6)->(10,10,16)->(5,5,16)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # Reshape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation function because it is implemented in the loss function
        return x


model = CNN().to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(epochs):
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device=device)
        targets = targets.to(device=device)

        pred = model(images)
        loss = criterion(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 2000 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{n_total_steps}], Loss : {loss.item():.3f}"
            )

print("Finished Training")
PATH = "F:\Pytorch\models"
torch.save(model.state_dict(), PATH)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc} %")

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f"Accuracy of {classes[i]}: {acc} %")
