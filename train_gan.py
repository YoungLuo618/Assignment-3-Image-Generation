import torch
from helper_lib.model import get_model
from helper_lib.trainer import get_mnist_loader, train_gan, save_gan
from helper_lib.generator import generate_samples, save_samples_grid

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    gan_model = get_model("GAN")

    loader = get_mnist_loader(batch_size=128, root="./data")

    gan_model = train_gan(
        gan_model,
        loader,
        device=device,
        epochs=10,        
        lr=2e-4,
        beta1=0.5
    )

    save_gan(gan_model, path="gan_trained.pth")

    imgs = generate_samples(gan_model, device=device, num_samples=16)
    save_samples_grid(imgs, outfile="gan_samples.png")

if __name__ == "__main__":
    main()
