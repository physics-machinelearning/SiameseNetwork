import numpy as np
import torch
from torch import nn
import torch.optim as optim

from model import ContrastiveLoss


class TrainConvNet:
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    def train(self, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.trainloader):
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    print('loss = ', running_loss/100)
                    running_loss = 0.0
            self.validation()

    def validation(self):
        with torch.no_grad():
            total = 0
            correct = 0
            for (inputs, labels) in self.testloader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy: ', 100*float(correct/total))


class TrainSiamese:
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        # self.criterion = nn.BCEWithLogitsLoss(size_average=True)
        self.criterion = ContrastiveLoss()
        # self.optimizer = optim.Adam(model.parameters(), lr = 0.001)
        self.optimizer = optim.Adadelta(model.parameters())

    def train(self, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data1, data2) in enumerate(self.trainloader):
                inputs1, labels1 = data1
                inputs2, labels2 = data2
                self.optimizer.zero_grad()

                out1, out2 = self.model(inputs1, inputs2)
                label = labels1 == labels2
                label = np.array(label, float)[:, np.newaxis]
                label = torch.from_numpy(label)
                loss = self.criterion(out1, out2, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print('loss = ', running_loss/100)
            running_loss = 0.0
            self.validation()

    def validation(self):
        with torch.no_grad():
            running_loss = 0.0
            for i, (data1, data2) in enumerate(self.testloader):
                inputs1, labels1 = data1
                inputs2, labels2 = data2
                self.optimizer.zero_grad()

                out1, out2 = self.model(inputs1, inputs2)
                label = labels1 == labels2
                label = np.array(label, float)[:, np.newaxis]
                label = torch.from_numpy(label)
                loss = self.criterion(out1, out2, label)

                running_loss += loss.item()

            print('validation loss = ', running_loss/10)
            running_loss = 0.0


if __name__ == '__main__':
    from model import ConvNet, Siamese
    from dataloader import loadmnist_siamese
    from train import TrainConvNet, TrainSiamese

    model = Siamese()
    trainloader, testloader = loadmnist_siamese(train_batch_size=100, test_batch_size=10)
    train_conv = TrainSiamese(model, trainloader, testloader)
    train_conv.train(100)
