from .gan import GAN

def get_model(model_name: str):
    model_name = model_name.upper()
    if model_name == "GAN":
        return GAN(latent_dim=100)
    else:
        raise ValueError(f"Unknown model_name {model_name}")
