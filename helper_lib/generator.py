import torch
import matplotlib.pyplot as plt
import math

def generate_samples(gan_model, device="cpu", num_samples=16):
    """
    Returns a tensor of generated images scaled to [0,1] for viewing.
    Shape: (num_samples, 1, 28, 28)
    """
    G = gan_model.generator.to(device)
    G.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, gan_model.latent_dim, device=device)
        fake_imgs = G(z)                    
        fake_imgs = (fake_imgs + 1) / 2.0   
    return fake_imgs.cpu()


def save_samples_grid(fake_imgs, outfile="samples.png"):
    """
    fake_imgs: (N,1,28,28) in [0,1]
    Saves a grid to outfile.
    """
    n = fake_imgs.shape[0]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(cols, rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        plt.imshow(fake_imgs[i,0].numpy(), cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved {outfile}")
