from cProfile import run
from turtle import shape
import PIL
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch


# check image
import matplotlib.pyplot as plt
from PIL import Image

# 신경망 구성
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16 * 5 * 5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        #print("num_features : "+str(num_features))
        return num_features

                        


net = NeuralNet()

#  손실함수 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)    

def model_training():

    mnist_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5,), std=(1.0,))])

    trainset = datasets.MNIST(root='./training/', train=True, download=True, transform=mnist_transform)
    testset = datasets.MNIST(root='./training/', train=False, download=True, transform=mnist_transform)

    train_loader = DataLoader(trainset, batch_size=16, shuffle=True)#, num_workers=2)
    test_loader = DataLoader(testset, batch_size=16, shuffle=False) #, num_workers=2)

    
    #  모델 학습
    total_batch = len(train_loader)
    print(total_batch)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader,0):
            inputs, labels = data
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 ==99:
                print('epoch: {}, iter: {}, loss: {}'.format(epoch+1,i+1,running_loss/2000))
                running_loss = 0.0 
                
    torch.save(net.state_dict(),'./mnist.pth')            

    # 모델 검증
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            
            net.load_state_dict(torch.load('./mnist.pth'))
            outputs = net(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print((100 * correct)/ total)



def upload_img_check(filepath):
    
    global net

    tmp_img = Image.open(filepath).convert('L')
    tmp_img = PIL.ImageOps.invert(tmp_img)
    tmp_imp = tmp_img.convert('1')
    resize = transforms.Resize((28,28))
    tmp_img = resize(tmp_img)
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(tmp_img)
    nomal_img = transforms.Normalize((0.5,), (1,))
    tensor = nomal_img(tensor)
    tensor = tensor.unsqueeze(0)

   
    with torch.no_grad():
        net.load_state_dict(torch.load('./mnist.pth'))    
        outputs = net(tensor)
        _, predicted = torch.max(outputs.data,1)
            
    return predicted.item()

