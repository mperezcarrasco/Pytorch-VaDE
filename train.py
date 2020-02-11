import math
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.utils.linear_assignment_ import linear_assignment

from models import Autoencoder, VaDE


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

class TrainerVaDE:
    """This is the trainer for the Variational Deep Embedding (VaDE).
    """
    def __init__(self, args, device, dataloader):
        self.autoencoder = Autoencoder().to(device)
        self.VaDE = VaDE().to(device)
        self.dataloader = dataloader
        self.device = device
        self.args = args


    def pretrain(self):
        """Here we train an stacked autoencoder which will be used as the initialization for the VaDE. 
        This initialization is usefull because reconstruction in VAEs would be weak at the begining
        and the models are likely to get stuck in local minima.
        """
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.002)
        self.autoencoder.apply(weights_init_normal) #intializing weights using normal distribution.
        self.autoencoder.train()
        print('Training the autoencoder...')
        for epoch in range(30):
            total_loss = 0
            for x, _ in self.dataloader:
                optimizer.zero_grad()
                x = x.to(self.device)
                x_hat = self.autoencoder(x)
                loss = F.binary_cross_entropy(x_hat, x, reduction='mean') # just reconstruction
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print('Training Autoencoder... Epoch: {}, Loss: {}'.format(epoch, total_loss))
        self.train_GMM() #training a GMM for initialize the VaDE
        self.save_weights_for_VaDE() #saving weights for the VaDE


    def train_GMM(self):
        """It is possible to fit a Gaussian Mixture Model (GMM) using the latent space 
        generated by the stacked autoencoder. This way, we generate an initialization for 
        the priors (pi, mu, var) of the VaDE model.
        """
        print('Fiting Gaussian Mixture Model...')
        x = torch.cat([data[0] for data in self.dataloader]).to(self.device) #all x samples.
        z = self.autoencoder.encode(x)
        self.gmm = GaussianMixture(n_components=10, covariance_type='diag')
        self.gmm.fit(z.cpu().detach().numpy())


    def save_weights_for_VaDE(self):
        """Saving the pretrained weights for the encoder, decoder, pi, mu, var.
        """
        print('Saving weights.')
        state_dict = self.autoencoder.state_dict()

        self.VaDE.load_state_dict(state_dict, strict=False)
        self.VaDE.pi_prior.data = torch.from_numpy(self.gmm.weights_).float().to(self.device)
        self.VaDE.mu_prior.data = torch.from_numpy(self.gmm.means_).float().to(self.device)
        self.VaDE.log_var_prior.data = torch.log(torch.from_numpy(self.gmm.covariances_)).float().to(self.device)
        torch.save(self.VaDE.state_dict(), self.args.pretrained_path)

    def train(self):
        """Training the VaDE
        """
        if self.args.pretrain==True:
            self.VaDE.load_state_dict(torch.load(self.args.pretrained_path,
                                                 map_location=self.device))
        else:
            self.VaDE.apply(weights_init_normal)
        self.optimizer = optim.Adam(self.VaDE.parameters(), lr=self.args.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=10, gamma=0.90)
        print('Training VaDE...')
        for epoch in range(self.args.epochs):
            self.train_VaDE(epoch)
            self.test_VaDE(epoch)
            lr_scheduler.step()


    def train_VaDE(self, epoch):
        self.VaDE.train()

        total_loss = 0
        for x, _ in self.dataloader:
            self.optimizer.zero_grad()
            x = x.to(self.device)
            x_hat, mu, log_var, z = self.VaDE(x)
            print('Before backward: {}'.format(self.VaDE.pi_prior))
            loss = self.compute_loss(x, x_hat, mu, log_var, z)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            print('After backward: {}'.format(self.VaDE.pi_prior))
        print('Training VaDE... Epoch: {}, Loss: {}'.format(epoch, total_loss/len(self.dataloader)))


    def test_VaDE(self, epoch):
        self.VaDE.eval()
        with torch.no_grad():
            total_loss = 0
            y_true, y_pred = [], []
            for x, true in self.dataloader:
                x = x.to(self.device)
                x_hat, mu, log_var, z = self.VaDE(x)
                gamma = self.compute_gamma(z, self.VaDE.pi_prior)
                pred = torch.argmax(gamma, dim=1)
                loss = self.compute_loss(x, x_hat, mu, log_var, z)
                total_loss += loss.item()
                y_true.extend(true.numpy())
                y_pred.extend(pred.cpu().detach().numpy())

            acc = self.cluster_acc(np.array(y_true), np.array(y_pred))
            print('Testing VaDE... Epoch: {}, Loss: {}, Acc: {}'.format(epoch, 
                  total_loss/len(self.dataloader), acc[0]))

    def compute_loss(self, x, x_hat, mu, log_var, z):
        p_c = self.VaDE.pi_prior
        gamma = self.compute_gamma(z, p_c)

        log_p_x_given_z = F.binary_cross_entropy(x_hat, x, reduction='sum')
        h = log_var.exp().unsqueeze(1) + (mu.unsqueeze(1) - self.VaDE.mu_prior).pow(2)
        h = torch.sum(self.VaDE.log_var_prior + h / self.VaDE.log_var_prior.exp(), dim=2)
        log_p_z_given_c = 0.5 * torch.sum(gamma * h)
        log_p_c = torch.sum(gamma * torch.log(p_c + 1e-9))
        log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-9))
        log_q_z_given_x = 0.5 * torch.sum(1 + log_var)

        loss = log_p_x_given_z + log_p_z_given_c - log_p_c +  log_q_c_given_x - log_q_z_given_x
        loss /= x.size(0)
        return loss

    def compute_gamma(self, z, p_c):
        h = (z.unsqueeze(1) - self.VaDE.mu_prior).pow(2) / self.VaDE.log_var_prior.exp()
        h += self.VaDE.log_var_prior
        h += torch.Tensor([np.log(np.pi*2)]).to(self.device)
        p_z_c = torch.exp(torch.log(p_c + 1e-9).unsqueeze(0) - 0.5 * torch.sum(h, dim=2))
        gamma = p_z_c / torch.sum(p_z_c, dim=1, keepdim=True)
        return gamma


    def cluster_acc(self, real, pred):
        D = max(pred.max(), real.max())+1
        w = np.zeros((D,D), dtype=np.int64)
        for i in range(pred.size):
            w[pred[i], real[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i,j] for i,j in ind])*1.0/pred.size*100, w


"""
    def compute_loss(self, x, x_hat, mu, log_var, z):
        p_c = self.VaDE.pi_prior
        gamma = self.compute_gamma(z, p_c)

        log_p_x_given_z = F.binary_cross_entropy(x_hat, x, reduction='sum') * x.size(1)
        log_p_z_given_c = 0.5 * torch.sum(gamma * torch.sum(mu.size(-1)*np.log(2*np.pi)\
                          + self.VaDE.log_var_prior.unsqueeze(0)\
                          + log_var.exp().unsqueeze(1)/self.VaDE.log_var_prior.exp()\
                          + (mu.unsqueeze(1) - self.VaDE.mu_prior).pow(2)/self.VaDE.log_var_prior.exp(), dim=2))
        log_p_c = torch.sum(gamma * torch.log(p_c + 1e-9))
        log_q_c_given_x = torch.sum(gamma * torch.log(gamma + 1e-9))
        log_q_z_given_x = 0.5 * torch.sum(1 + log_var)

        loss = log_p_x_given_z + log_p_z_given_c - log_p_c + log_q_c_given_x - log_q_z_given_x
        loss /= x.size(0)

        return loss

    def compute_gamma(self, z, p_c):
        h = (z.unsqueeze(1) - self.VaDE.mu_prior).pow(2) / self.VaDE.log_var_prior.exp() + \
             np.log(2*np.pi) + self.VaDE.log_var_prior
        p_z_c = torch.exp(torch.log(p_c + 1e-9).unsqueeze(0) - 0.5 * torch.sum(h, dim=2))
        gamma = p_z_c / torch.sum(p_z_c + 1e-9, dim=1, keepdim=True)
        return gamma
"""


