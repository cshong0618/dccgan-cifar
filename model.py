import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision


class D(nn.Module):
    def __init__(self, input_channels=3, output_size=10):
        super(D, self).__init__()

        self.input_channels = input_channels
        self.output_size = output_size

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=self.input_channels, out_channels=8, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.02, inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout2d(p=0.5)
        )

        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.02, inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout2d(p=0.5)
        )

        self.conv3 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02, inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout2d(p=0.5)
        )

        self.fc_out = nn.Sequential(
            nn.Linear(16 * 11 * 11, 120),
            nn.Linear(120, 64),
            nn.Linear(64, self.output_size),
        )

    def forward(self, X):
        out = self.conv1(X)         # [batch_size, 32, 18, 18]
        out = self.conv2(out)       # [batch_size, 64, 11, 11]
        #out = self.conv3(out)       # [batch_size, 128, 8, 8]

        out = out.view(out.size(0), -1)
        out = self.fc_out(out)
        return out

class D2(nn.Module):
    def __init__(self, input_channels=3, output_size=10):
        super(D2, self).__init__()
        self.input_channels=input_channels
        self.output_size = output_size

        self.conv1 = nn.Conv2d(self.input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.output_size)

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, self.output_size)
        )

    def forward(self, X):
        x = self.features(X)
        x = x.view(-1, 16 * 5 * 5)
        x = self.classifier(x)
        return x

def resnet(out_features=10):
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, out_features)
    return model_conv

class TransferD(nn.Module):
    def __init__(self, features, out_features=11):
        super(TransferD, self).__init__()
        self.out_features = out_features
        self.features = features

        for params in self.features.parameters():
            params.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 11),
        )

    def forward(self, X):
        x = self.features(X)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x 

def vgg16(out_features=11):
    vgg = torchvision.models.vgg16(pretrained=True)

    _d = TransferD(vgg.features, 11)
    return _d

class CCNN(nn.Module):
    def __init__(self, input_class=10, dimen=(32, 32, 3)):
        super(CCNN, self).__init__()
        self.input_class = input_class
        self.w = dimen[0]
        self.h = dimen[1]
        self.d = dimen[2]

        self.fc_in = nn.Sequential(
            #nn.Linear(input_class, self.w * self.h * self.d)
            nn.Linear(input_class, 128),
            nn.Sigmoid(),
            nn.Linear(128, self.w * self.h * self.d)
        )

        self.decoder = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels = self.d, out_channels=32,kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02, inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout2d(0.5),
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels = 32, out_channels=64,kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02, inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout2d(0.5),
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels = 64, out_channels=128,kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(128),            
            nn.LeakyReLU(0.02, inplace=True),
            nn.AvgPool2d(2),
            nn.Dropout2d(0.5)
        ) # [batch_size, 128, 8, 8]

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.02, inplace=True),
            nn.ConvTranspose2d(16, self.d, kernel_size=4, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, X, noise, latent_code=0):
        fc_out = self.fc_in(X)
        fc_out = fc_out.view(fc_out.size(0), self.d, self.w, self.h)
        noise = fc_out + noise
        decoded = self.decoder(noise)
        out = self.encoder(decoded)

        return out
