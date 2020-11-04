#main training
from models.dataset_generator import generate_data
from models.network_model import SupervisedModel

import torch
import numpy as np

input_size = 1
output_size = 4
mymodel = SupervisedModel(input_size, output_size)





if __name__ == "main":

    epochs = 10
    for e in range(epochs):
        pass

