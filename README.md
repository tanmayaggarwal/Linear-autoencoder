# Linear-autoencoder

Overview:

This code builds a linear autoencoder that takes images from the MNIST dataset as input (size: 28 x 28), creates a compressed
representation, and then decodes it back to the original image.

Additional details:

The images from the MNIST dataset are already normalized such that the values are between 0 and 1. The encoder and the decoder are made of one linear layer. We use a sigmoid activation on the output layer to get values that match the input value range.
All other layers have a reLU activation function.

List of dependencies is as follows:

import torch <br/>
import numpy as np <br/>
from torchvision import datasets <br/>
import torchvision.transforms as transforms <br/>
import matplotlib.pyplot as plt <br/>
import torch.nn as nn <br/>
import torch.nn.functional as F <br/>
