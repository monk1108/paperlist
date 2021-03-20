# Fully convolutional Networks for Semantic Segmentation
# cifar10 has 10 categories of images, 32*32, and has 3 channels(RGB).
# 写完才发现FCN是语义分割，在cifar上跑干什么。。

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.models.vgg import VGG

vgg16_layers = [64, 64, 'm', 128, 128, 'm', 256, 256, 256, 'm', 512, 512, 512, 'm', \
                512, 512, 512, 'm']

def make_blocks(layers):  # building the downconv blocks
    downlayers = []
    input_layer = 3
    for layer in layers:
        if layer == 'm':
            downlayers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            downlayers.append(nn.Conv2d(input_layer, layer, kernel_size=3, padding=1))
            downlayers.append(nn.ReLU(inplace=True))
            input_layer = layer
    return nn.Sequential(*downlayers)

class VGG16(VGG):    # 注意要继承父类VGG
    def __init__(self):
        super().__init__(make_blocks(vgg16_layers))

    def forward(self, x):
        vgg16_blocks = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31))
        outmax_dict = {}
        for b in range(len(vgg16_blocks)):
            for i in range(vgg16_blocks[b][0], vgg16_blocks[b][1]):
                x = self.features[i](x)
            outmax_dict["x%d" % (b+1)] = x

        return outmax_dict


class FCN32(nn.Module):
    def __init__(self, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.upconv5 = nn.ConvTranspose2d(64, 32 ,kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.upconv6 = nn.ConvTranspose2d(32, 10, kernel_size=1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.drop1 = nn.Dropout2d()
        self.drop2 = nn.Dropout2d()

    def forward(self, input):
        output = self.pretrained_net(input)    # output is a dict containing 5 results of maxpooling
        output5 = output['x5']
        output = self.bn1(self.relu(self.upconv1(output5)))
        output = self.bn2(self.relu(self.upconv2(output)))
        output = self.bn3(self.relu(self.upconv3(output)))
        output = self.bn4(self.relu(self.upconv4(output)))
        output = self.bn5(self.relu(self.upconv5(output)))
        output = self.bn6(self.relu(self.upconv6(output)))
        output = output.view(4, 10 * 32 * 32)
        output = self.drop1(self.fc1(output))
        output = self.drop2(self.fc2(output))
        output = self.fc3(output)


        return output
        
class FCN16(nn.Module):
    def __init__(self, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.upconv5 = nn.ConvTranspose2d(64, 32 ,kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.upconv6 = nn.ConvTranspose2d(32, 10, kernel_size=1)
        self.fc1 = nn.Linear(10*32*32, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.drop1 = nn.Dropout2d()
        self.drop2 = nn.Dropout2d()

    def forward(self, input):
        output = self.pretrained_net(input)    # output is a dict containing 5 results of maxpooling
        output5 = output['x5']   # [4, 512, 1, 1] ?
        output4 = output['x4']   # [4, 512, 2, 2]
        output = self.bn1(self.relu(self.upconv1(output5)))   # [4, 512, 2, 2]
        output = self.bn2(self.relu(self.upconv2(output + output4)))
        # output = self.bn2(output + output4)
        output = self.bn3(self.relu(self.upconv3(output)))
        output = self.bn4(self.relu(self.upconv4(output)))
        output = self.bn5(self.relu(self.upconv5(output)))
        output = self.upconv6(output)   # [4, 10, 32, 32]
        output = output.view(4, 10*32*32)
        output = self.drop1(self.fc1(output))
        output = self.drop2(self.fc2(output))
        output = self.fc3(output)

        return output


class FCN8(nn.Module):
    def __init__(self, pretrained_net):
        super().__init__()
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.upconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.upconv5 = nn.ConvTranspose2d(64, 32 ,kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.upconv6 = nn.ConvTranspose2d(32, 10, kernel_size=1)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 10)
        self.drop1 = nn.Dropout2d()
        self.drop2 = nn.Dropout2d()

    def forward(self, input):
        output = self.pretrained_net(input)    # output is a dict containing 5 results of maxpooling
        output5 = output['x5']
        output4 = output['x4']
        output3 = output['x3']
        output = self.bn1(self.relu(self.upconv1(output5)))
        output = self.relu(self.upconv2(output))
        output = self.bn2(output + output4)
        output = self.relu(self.upconv3(output))
        output = self.bn3(output + output3)
        output = self.bn4(self.relu(self.upconv4(output)))
        output = self.bn5(self.relu(self.upconv5(output)))
        output = self.bn6(self.relu(self.upconv6(output)))
        output = output.view(4, 10 * 32 * 32)
        output = self.drop1(self.fc1(output))
        output = self.drop2(self.fc2(output))
        output = self.fc3(output)

        return output


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# def Dice(predict, label):
#     _, predict = predict.max(1)
#     output = 0
#     for i in range(predict.shape[0]):    # the dice function
#         output += 2 * (predict[i, :, :] * label[i, :, :]).sum().sum().cpu().numpy() / (
#                     predict[i, :, :].sum().sum().cpu().numpy() + label[i, :, :].sum().sum().cpu().numpy())
#     return output

def one_hot(x):
	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
	return torch.eye(10)[x,:]


if __name__ == "__main__":
    batchsize = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                             shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    vgg_model = VGG16()
    vgg_model.to(device)
    fcn_model = FCN16(pretrained_net=vgg_model)
    fcn_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-3, momentum=0.9)
    num_of_epochs = 1500

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # if choice == 'Train':
    for epoch in range(num_of_epochs):
        print(epoch)
        fcn_model.train()  # train mode
        trainloss = 0
        # traindice = 0
        optimizer.zero_grad()
        for i, data in enumerate(trainloader):  # one batch
            images, labelings = data
            # images: [4, 3, 32, 32], labelings:[4]
            if torch.cuda.is_available():
                images = images.cuda().float()
                labelings = labelings.cuda().long()
            output = fcn_model(images)
            output = nn.functional.sigmoid(output)
            # _, predicted = torch.max(output, 1)
            # ohpre = one_hot(predicted)
            ohlab = one_hot(labelings)
            loss = criterion(output, labelings)
            loss.backward()
            if loss == 'nan':
                print(i)

            optimizer.step()  # update the params
            # trainloss += loss.item()
            # traindice += Dice(output, ohlab)
            trainloss += loss

        print('loss:%5f' % trainloss)