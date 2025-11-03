Assignment 3: Image Generation

Jianyang Luo

This project implements a Generative Adversarial Network (GAN) using PyTorch to generate hand-written digits from the MNIST dataset, and exposes the result through a FastAPI REST API.

__

Descriptions:

gan.py # Generator, Discriminator, GAN wrapper 

model.py # get_model("GAN") factory 

trainer.py # train_gan(), save_gan(), load_gan() 

generator.py # generate_samples(), save_samples_grid() 

main.py # FastAPI endpoints /gan/samples & /gan/grid 

train_gan.py # Train GAN and save gan_trained.pth & gan_samples.png 

requirements.txt # Dependencies 

README.md # Project description 

__

1.Install Dependencies:

cd gan_project
pip install -r requirements.txt

2️.Train the GAN 

Run:
python train_gan.py

This will download MNIST dataset and train the GAN model


We can save:

gan_trained.pth 

gan_samples.png:This the generated 4×4 grid of hand-written digits



3.Run the API 

Start the FastAPI server:
uvicorn api.main:app --reload

When you see:
INFO: Uvicorn running on http://127.0.0.1:8000 and INFO: Application startup complete.

The API is running successfully.

__

Available Endpoints
/gan/samples?n=4


Returns generated digits as pixel arrays (JSON).


Another Endpoints
/gan/grid

Returns the generated image file gan_samples.png directly.


Access in browser 
http://127.0.0.1:8000/gan/grid

We’ll see the 4×4 grid of GAN-generated digits.

———

We can also run this project using Docker. 

From the project root directory, build the image:

docker build -t gan_api .


Then run the container:

docker run -p 8000:8000 gan_api

This will start the FastAPI service on port 8000.
Open browser at http://127.0.0.1:8000/gan/grid to see the generated handwritten digit

The Dockerfile is automatically installs dependencies from requirements.txt and then launches the app with uvicorn api.main:app --host 0.0.0.0 --port 8000.





