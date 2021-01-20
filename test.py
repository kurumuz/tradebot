import cv2
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
#read image

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
PATH = './mnist.pth'
net.load_state_dict(torch.load(PATH))




img = cv2.imread('test.png')

#grayscale
img = cv2.resize(img ,None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.waitKey(0)

#binarize 
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = gray[y:y+h, x:x+w]
    roi = cv2.resize(roi, (28, 28), interpolation = cv2.INTER_AREA)
    transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    roitensor = transformation(roi).float()
    roitensor = roitensor.unsqueeze_(0)

    with torch.no_grad():
        output = net(roitensor)
        _, predicted = torch.max(output.data, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    print(predicted.item())
    #cv2.putText(img,predicted,(x,y), font, 4,(255,255,255),2,cv2.LINE_AA)
    # show ROI
    #cv2.imwrite('roi_imgs.png', roi)
    cv2.imshow("test", roi)
    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
    #fuck = (x, y, )
    cv2.waitKey(0)

cv2.imshow('marked areas',img)
cv2.waitKey(0)