import torch
import torchvision
from torch import nn
from torch import optim

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

train_data = datasets.MNIST("./", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("./", train=False, download=True, transform=transforms.ToTensor())

batch_size = 64

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

images, target = iter(train_loader).next()

print(images.data.shape)

image_grid = torchvision.utils.make_grid(images, nrows=8)

plt.imshow(image_grid.permute(1, 2, 0))

print(target.data.reshape(8, 8))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)

        return self.softmax(x)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(560, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        bs = x.shape[0]
        x = self.relu(self.pool(self.batchNorm1(self.conv1(x))))
        x = self.dropout(x)
        x = self.relu(self.pool(self.batchNorm2(self.conv2(x))))

        x = x.view(bs, -1)
        print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return self.sofmax(x)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)

model = Net()
model.to(device)

learning_rate = 0.001

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

CELoss = nn.CrossEntropyLoss()

train_losses = []
test_losses = []


def train(epoch):
    model.train()

    for batch_idx, (images, target) in enumerate(train_loader):
        images, target = images.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = CELoss(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Train Epoch {} [{} / 60000] \t Loss: {}".format(epoch, batch_idx * len(images), loss.item()))
            train_losses.append(loss.item())


def test():
    model.eval()

    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for (images, target) in test_loader:
            images, target = images.to(device), target.to(device)

            output = model(images)
            loss = CELoss(output, target)
            test_loss += loss.item()

            test_losses.append(test_loss / len(test_loader.dataset))

            predicts = torch.argmax(output, 1)
            correct += (predicts == target).sum()

        print("Test Avg. Loss: {} \t Accuracy: {}".format(test_loss,  correct / len(test_loader.dataset)))
        test_losses.append(test_loss)


n_epochs = 30

for epoch in range(1, n_epochs+1):
    train(epoch)
    test()




