#model class

import torch
from torch import nn
import numpy as np

class SupervisedModel():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.hidden1 = nn.Linear(self.input_size, 8)
        self.hidden2 = nn.Linear(8, 16)
        self.hidden3 = nn.Linear(16, self.output_size)

        self.activation_hidden = nn.ReLU()
        self.activation_final = nn.Softmax()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.activation_hidden(x)

        x = self.hidden2(x)
        x = self.activation_hidden(x)

        x = self.hidden3(x)
        x = self.activation_final(x)

        return x

