import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import numpy
import random

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

'''
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
'''

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
#net.load_state_dict(torch.load("charb.pth"))
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


imagesx = []

for x in range(0, 30):
    imagesx.append(cv2.imread('allah/' + str(x) + ".png", 0))

imagesmore = []

for i in range(0, 33):
    for img in imagesx:
        imagesmore.append(img) 

print(len(imagesmore))

f = open('label.txt')
labelsx = f.read().split('\n')

labelsmore = []
for i in range(0, 33):
    for label in labelsx:
        labelsmore.append(label) 

print(len(labelsx))

mapIndexPosition = list(zip(imagesmore, labelsmore))
random.shuffle(mapIndexPosition)
imagesmore, labelsmore = zip(*mapIndexPosition)

''''
tlabels = []
for i in range(0, len(labels)):
    label_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    label_list[int(labels[i])] = 1
    b = torch.FloatTensor(label_list)
    tlabels.append(b)
'''

tlabels = []
for i in range(0, len(labelsmore)):
    b = torch.tensor([int(labelsmore[i])], dtype=torch.long)
    tlabels.append(b)

#for xak in tlabels:
#    print(xak)

timages = []
transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])



for i in range(0, len(imagesmore)):
    timages.append(transformation(imagesmore[i]).float().unsqueeze_(0))

'''
for i in range(0, 40):
    print("f", labelsmore[i])
    cv2.imshow("f", imagesmore[i])
    cv2.waitKey(0)
'''

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(0, len(timages)):
        #print("test")
        # get the inputs; data is a list of [inputs, labels]
        inputx, labelx = timages[i].to(device), tlabels[i].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputx = net(inputx)
        #print(outputx)
        loss = criterion(outputx, labelx)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 900 == 899:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 899))
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), "charbb.pth")
'''
PATH = './mnist.pth'
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if total == 0:
            print(images)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
'''


