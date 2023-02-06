# VAE and GAN variants

This repository experiments with VAE, GAN and VAE/GAN architectures on the Fashion MNIST dataset.

## How to use it

Currently only AutoEncoder is implemented, while possible to add more features in the future, 
it can be used without issue. 

To train an AutoEncoder, run the code below. 
```
python ae.py --train 
python ae.py --train --model-path my_ae.pth
python ae.py --train --model-path my_ae.pth --seed 100
```
If model path is not given, model will be saved in 
`./models/best_ae.pth`, else it will be saved
in `./models/{provided_path}`. 
Training will also generate a png image showing metrics. 
You may provide seed to ensure reproducibility.

Our evaluation takes 10 random images from the validation dataset,
then plot them against their reconstructions. 
Provide model path only. 
Use seed to ensure reproducibility.
```
python ae.py --model-path my_ae.pth
python ae.py --model-path my_ae.pth --seed 100
```
![Images reconstructed by a trained autoencoder.](./media/image_compare.png)