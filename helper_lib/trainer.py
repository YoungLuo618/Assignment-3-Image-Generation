import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_loader(batch_size=128, root="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),                
        transforms.Normalize((0.5,), (0.5,))  
    ])
    trainset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )
    loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return loader


def train_gan(
    gan_model,
    data_loader,
    device="cpu",
    epochs=10,
    lr=2e-4,
    beta1=0.5
):
    """
    gan_model: GAN() with .generator and .discriminator
    data_loader: MNIST loader
    device: 'cpu' or 'cuda'
    """

    G = gan_model.generator.to(device)
    D = gan_model.discriminator.to(device)

    criterion = nn.BCELoss()

    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

    latent_dim = gan_model.latent_dim

    for epoch in range(epochs):
        for real_imgs, _ in data_loader:

            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            D.train()
            opt_d.zero_grad()

            d_real = D(real_imgs)                   
            loss_real = criterion(d_real, real_labels)

            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = G(z)                        
            d_fake = D(fake_imgs.detach())
            loss_fake = criterion(d_fake, fake_labels)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()

            d_fake_for_g = D(fake_imgs)            
            g_loss = criterion(d_fake_for_g, real_labels)

            g_loss.backward()
            opt_g.step()

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}"
        )

    return gan_model


def save_gan(gan_model, path="gan_trained.pth"):
    torch.save({
        "generator_state_dict": gan_model.generator.state_dict(),
        "discriminator_state_dict": gan_model.discriminator.state_dict(),
        "latent_dim": gan_model.latent_dim
    }, path)


def load_gan(gan_model, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    gan_model.generator.load_state_dict(checkpoint["generator_state_dict"])
    gan_model.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    gan_model.latent_dim = checkpoint.get("latent_dim", gan_model.latent_dim)
    return gan_model
