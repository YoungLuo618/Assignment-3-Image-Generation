import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, z):
        x = self.fc(z)                        # (B, 128*7*7)
        x = x.view(-1, 128, 7, 7)             # (B,128,7,7)

        x = self.deconv1(x)                   # (B,64,14,14)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.deconv2(x)                   # (B,1,28,28)
        x = self.tanh(x)                      # [-1,1]
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * 7 * 7, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)             
        x = self.lrelu(x)

        x = self.conv2(x)            
        x = self.bn2(x)
        x = self.lrelu(x)

        x = x.view(x.size(0), -1)     
        x = self.fc(x)                
        x = self.sigmoid(x)           
        return x


class GAN(nn.Module):
    """
    Just a convenience wrapper so we can pass one object around.
    """
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()
        self.latent_dim = latent_dim

    def sample_latent(self, batch_size, device):
        return torch.randn(batch_size, self.latent_dim, device=device)
