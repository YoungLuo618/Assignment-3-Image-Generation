from fastapi import FastAPI
from fastapi.responses import FileResponse
import torch

from helper_lib.model import get_model
from helper_lib.trainer import load_gan
from helper_lib.generator import generate_samples

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

gan_model = get_model("GAN")
try:
    gan_model = load_gan(gan_model, path="gan_trained.pth", device=device)
except FileNotFoundError:
    pass

@app.get("/gan/samples")
def get_gan_samples(n: int = 16):
    imgs = generate_samples(gan_model, device=device, num_samples=n)
    imgs_list = imgs.squeeze(1).numpy().tolist()
    return {
        "num_images": n,
        "images": imgs_list
    }

@app.get("/gan/grid")
def get_gan_grid():
    return FileResponse("gan_samples.png", media_type="image/png")
