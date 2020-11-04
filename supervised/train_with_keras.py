#main training script
import os
import numpy as np
from random import randint
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import to_categorical

from childs.dataset_generator import generate_data
from childs.network_model_keras import KerasNetwork
from childs.predict_keras import myPredict

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#prepare data
sample_number = 1000
samples = generate_data(sample_number)
x = samples[:,0]
y = samples[:,1]

y_onehot = to_categorical(y, num_classes=4) #one-hot encoding, y train(labels)

#optimizers
optimizer1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
optimizer2 = SGD(lr=0.1, momentum=0.1)

input_dim = 1
output_dim = 4

mynet = KerasNetwork(input_dim, output_dim)

#model compile
mynet.model.compile(loss = "categorical_crossentropy", optimizer = optimizer1, metrics=["accuracy"])

#train
history = mynet.model.fit(x = x, y = y_onehot, batch_size = 64, epochs = 300, verbose = 1)

print("training completed")
#logs
plt.plot(history.history['loss'], color='b', label="loss")
plt.title("CE loss vs. epochs")
plt.xlabel("epochs")
plt.ylabel("CE loss")
plt.legend()
plt.show()

#prediction
x_test = np.array([150])
myPredict(x_test, mynet.model)





