import torch
import torch.nn as nn

class D(nn.Module):
    def __init__(self, input_channels=3, output_size=10):
        super(D, self).__init__()

        self.input_channels = input_channels
        self.output_size = output_size

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.5)
        )

        self.conv2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.5)
        )

        self.conv3 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.5)
        )

        self.fc_out = nn.Sequential(
            nn.Linear(128 * 8 * 8, 100),
            nn.ReLU(),
            nn.Linear(100, self.output_size),
            nn.Softmax()
        )

    def forward(self, X):
        out = self.conv1(X)         # [batch_size, 32, 18, 18]
        out = self.conv2(out)       # [batch_size, 64, 11, 11]
        out = self.conv3(out)       # [batch_size, 128, 8, 8]

        out = out.view(out.size(0), -1)
        out = self.fc_out(out)
        return out

class CCNN(nn.Module):
    def __init__(self, input_class=10, dimen=(32, 32, 3)):
        super(CCNN, self).__init__()
        self.input_class = input_class
        self.w = dimen[0]
        self.h = dimen[1]
        self.d = dimen[2]

        self.fc_in = nn.Sequential(
            nn.Linear(input_class, self.w * self.h * self.d)
        )

        self.decoder = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels = self.d, out_channels=32,kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Dropout2d(0.5),
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels = 32, out_channels=64,kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Dropout2d(0.5),
            nn.ZeroPad2d(2),
            nn.Conv2d(in_channels = 64, out_channels=128,kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Dropout2d(0.5)
        ) # [batch_size, 128, 8, 8]

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(),
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
