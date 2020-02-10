import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class VaDE(nn.Module):
    def __init__(self, conv_dim=64, latent_dim=256, n_classes=31):
        super(VaDE, self).__init__()
        self.pi_prior = Parameter(torch.ones(n_classes)/n_classes)
        self.mu_prior = Parameter(torch.randn(n_classes, latent_dim))
        self.log_var_prior = Parameter(torch.randn(n_classes, latent_dim))

        self.n_classes = n_classes
        self.cnn4 = Conv(conv_dim*6, conv_dim*6, 3, 1, 1, groups=2, bn=False)
        self.cnn5 = Conv(conv_dim*6, conv_dim*4, 3, 1, 1, groups=2, bn=False)
        self.lin0 = nn.Linear(256 * 6 * 6, 4096)
        self.lin1 = nn.Linear(4096, 4096)

        self.fc1 = nn.Linear(4096, latent_dim)
        self.fc2 = nn.Linear(4096, latent_dim)

        self.lin = nn.Linear(latent_dim, 512)
        self.cnn6 = Deconv(conv_dim*8, conv_dim*4, 6, 6, 0, bn=False)
        self.cnn7 = Deconv(conv_dim*4, conv_dim*6, 3, 2, 1, bn=False)
        self.cnn8 = Deconv(conv_dim*6, conv_dim*6, 3, 1, 0, bn=False)

    def encode(self, x):
        h = F.relu(self.cnn4(x))
        h = F.max_pool2d((F.relu(self.cnn5(h))), kernel_size=3, stride=2)
        h = h.view(-1, 256 * 6 * 6)
        h = F.dropout(F.relu(self.lin0(h)), p=0.5) #dropout
        h = F.dropout(F.relu(self.lin1(h)), p=0.5)
        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        #print (eps)
        return mu + eps * std

    def decode(self, z):
        #h = torch.cat((z, code), dim = 1)
        h = self.lin(z)
        h = h.view(-1, 512, 1, 1)
        h = F.relu(self.cnn6(h))
        h = F.relu(self.cnn7(h))
        return F.relu(self.cnn8(h))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var, z

class Autoencoder(nn.Module):
    def __init__(self, conv_dim=64, latent_dim=256):
        super(Autoencoder, self).__init__()
        self.cnn4 = Conv(conv_dim*6, conv_dim*6, 3, 1, 1, groups=2, bn=False)
        self.cnn5 = Conv(conv_dim*6, conv_dim*4, 3, 1, 1, groups=2, bn=False)
        self.lin0 = nn.Linear(256 * 6 * 6, 4096)
        self.lin1 = nn.Linear(4096, 4096)

        self.fc1 = nn.Linear(4096, latent_dim)

        self.lin = nn.Linear(latent_dim, 512)
        self.cnn6 = Deconv(conv_dim*8, conv_dim*4, 6, 6, 0, bn=False)
        self.cnn7 = Deconv(conv_dim*4, conv_dim*6, 3, 2, 1, bn=False)
        self.cnn8 = Deconv(conv_dim*6, conv_dim*6, 3, 1, 0, bn=False)

    def encode(self, x):
        h = F.relu(self.cnn4(x))
        h = F.max_pool2d((F.relu(self.cnn5(h))), kernel_size=3, stride=2)
        h = h.view(-1, 256 * 6 * 6)
        h = F.dropout(F.relu(self.lin0(h)), p=0.5) #dropout
        h = F.dropout(F.relu(self.lin1(h)), p=0.5)
        return self.fc1(h)

    def decode(self, z):
        #h = torch.cat((z, code), dim = 1)
        h = self.lin(z)
        h = h.view(-1, 512, 1, 1)
        h = F.relu(self.cnn6(h))
        h = F.relu(self.cnn7(h))
        return F.relu(self.cnn8(h))

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)
        return x_reconst


class feature_extractor(nn.Module):
    def __init__(self, conv_dim=96):
        super(feature_extractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=1),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(size=5, alpha=1e-04, beta=0.75, k=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )
    def forward(self, x):
        return self.features(x)


class Conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bn=True, groups=1):
        super(Conv, self).__init__()
        self.bn = bn
        self.conv2d = nn.Conv2d(in_channels=dim_in, out_channels= dim_out,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=True, groups=groups)
        self.bn2d = nn.BatchNorm2d(num_features=dim_out)
    def forward(self, x):
        if self.bn:
            return self.bn2d(self.conv2d(x))
        else:
            return self.conv2d(x)


class Deconv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, bn=True):
        super(Deconv, self).__init__()
        self.bn = bn
        self.deconv2d = nn.ConvTranspose2d(in_channels=dim_in, out_channels=dim_out, 
                                           kernel_size=kernel_size, stride=stride, 
                                           padding=padding, bias=True) 
        self.bn2d = nn.BatchNorm2d(num_features=dim_out)
    def forward(self, x):
        if self.bn:
            return self.bn2d(self.deconv2d(x))
        else: 
            return self.deconv2d(x)
