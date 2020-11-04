#main training
from models.dataset_generator import generate_data
from models.network_model import Network

import torch
from torch import nn, optim
import numpy as np


if __name__ == "__main__":

    sample_number = 10
    input_size = 1
    output_size = 4

    samples = generate_data(sample_number)
    samples = torch.tensor(samples)

    model = Network(input_size, output_size)
    objective = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for e in range(epochs):
        print("epoch:",e)

        for sample in samples:
            x = sample[0], y_true = sample[1]
            y_pred = model.forward(x)
            #y_pred = torch.squeeze(y_pred)
            loss = objective(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

