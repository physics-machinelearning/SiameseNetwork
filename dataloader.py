import numpy as np
import random
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms


def loadmnist(train_batch_size, test_batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, 
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=True, num_workers=2)
    return trainloader, testloader


def smallmnist(train_batch_size, test_batch_size):
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home='./data_sklearn')
    x = np.array(x, np.float32)
    x = x.reshape((x.shape[0], 28, 28))
    x = x[:,np.newaxis,:,:]

    x /= np.max(x)
    y = np.array(y, np.int64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True)
    train_size = test_size = 1000
    class_num = 10
    i_size = train_size // class_num

    indexes = []
    for i in range(class_num):
        index_i = np.where(y_train==i)[0].flatten()
        index = np.random.choice(len(index_i), size=i_size, replace=False)
        index_i = index_i[index]
        indexes.append(index_i)
    indexes = np.array(indexes).flatten()

    x_train = x_train[indexes]
    y_train = y_train[indexes]

    indexes = []
    for i in range(class_num):
        index_i = np.where(y_test==i)[0].flatten()
        index = np.random.choice(len(index_i), size=i_size, replace=False)
        index_i = index_i[index]
        indexes.append(index_i)
    indexes = np.array(indexes).flatten()

    x_test = x_test[indexes]
    y_test = y_test[indexes]

    x_train_tensor = torch.stack([torch.from_numpy(np.array(i)) for i in x_train])
    y_train_tensor = torch.stack([torch.from_numpy(np.array(i)) for i in y_train])

    x_test_tensor = torch.stack([torch.from_numpy(np.array(i)) for i in x_test])
    y_test_tensor = torch.stack([torch.from_numpy(np.array(i)) for i in y_test])

    train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    train_loader1 = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    train_loader2 = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    return train_loader1, train_loader2, test_loader


def loadmnist_siamese(train_batch_size, test_batch_size):
    trainset = SiameseTrainDataset()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=2)
    testset = SiameseTestDataset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=True, num_workers=2)
    return trainloader, testloader


class SiameseTrainDataset(Dataset):
    def __init__(self, times=200, way=5):
        super(SiameseTrainDataset, self).__init__()
        self.x, self.y = self._load()
        self.times = times
        self.way = way
        self.num_classes = 10

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        if idx == 0:
            c1 = random.randint(0, self.num_classes-1)
            c2 = c1
            imgs1 = random.choice(self.x[self.y==c1])
            imgs2 = random.choice(self.x[self.y==c1])
        else:
            c1 = random.randint(0, self.num_classes - 1)
            c2 = random.randint(0, self.num_classes - 1)
            while c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            imgs1 = random.choice(self.x[self.y==c1])
            imgs2 = random.choice(self.x[self.y==c2])
        data1 = (imgs1, c1)
        data2 = (imgs2, c2)
        return data1, data2

    def _load(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
        trainset = torchvision.datasets.MNIST(root='./data', train=True, 
                                              download=True, transform=transform)
        return trainset.data[:, None, :, :], trainset.targets


class SiameseTestDataset(Dataset):
    def __init__(self, times=200, way=5):
        super(SiameseTestDataset, self).__init__()
        self.x, self.y = self._load()
        self.times = times
        self.way = way
        self.num_classes = 10

    def __len__(self):
        return self.times * self.way

    def __getitem__(self, index):
        idx = index % self.way
        if idx == 0:
            c1 = random.randint(0, self.num_classes-1)
            c2 = c1
            imgs1 = random.choice(self.x[self.y==c1])
            imgs2 = random.choice(self.x[self.y==c1])
        else:
            c1 = random.randint(0, self.num_classes - 1)
            c2 = random.randint(0, self.num_classes - 1)
            while c1 == c2:
                c2 = random.randint(0, self.num_classes - 1)
            imgs1 = random.choice(self.x[self.y==c1])
            imgs2 = random.choice(self.x[self.y==c2])
        data1 = (imgs1, c1)
        data2 = (imgs2, c2)
        return data1, data2

    def _load(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )
        testset = torchvision.datasets.MNIST(root='./data', train=True, 
                                             download=True, transform=transform)
        return testset.data[:, None, :, :], testset.targets
