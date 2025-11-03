GAN on MNIST – Assignment 3

Jianyang Luo

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate hand-written digits from the MNIST dataset, and exposes the result through a FastAPI REST API.


gan_project/
├─ helper_lib/
│ ├─ gan.py # Generator, Discriminator, GAN wrapper 
│ ├─ model.py # get_model("GAN") factory 
│ ├─ trainer.py # train_gan(), save_gan(), load_gan() 
│ ├─ generator.py # generate_samples(), save_samples_grid() 
│ └─ init.py
│
├─ api/
│ ├─ main.py # FastAPI endpoints /gan/samples & /gan/grid 
│ └─ init.py
│
├─ train_gan.py # Train GAN and save gan_trained.pth & gan_samples.png 
├─ requirements.txt # Dependencies 
└─ README.md # Project description 


1.Install Dependencies:

cd gan_project
pip install -r requirements.txt

requirements.txt
torch
torchvision
matplotlib
fastapi
uvicorn


2️.Train the GAN 

Run:
python train_gan.py

This will:

Download MNIST dataset

Train the GAN model

Save:

gan_trained.pth 

gan_samples.png:This the generated 4×4 grid of hand-written digits



3.Run the API 

Start the FastAPI server:
uvicorn api.main:app --reload

When you see:
INFO: Uvicorn running on http://127.0.0.1:8000

INFO: Application startup complete.
The API is running successfully.


Available Endpoints
/gan/samples?n=4

Returns generated digits as pixel arrays (JSON).


/gan/grid

Returns the generated image file gan_samples.png directly.


Access in your browser 
http://127.0.0.1:8000/gan/grid

You’ll see the 4×4 grid of GAN-generated digits.




