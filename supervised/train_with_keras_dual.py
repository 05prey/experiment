#main training script
import os
import numpy as np
from random import randint
import matplotlib.pyplot as plt

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import to_categorical

#from childs.dataset_generator import generate_data
from childs_1.network_model_keras import KerasNetwork_1
from childs_1.predict_keras import myPredict_1

#from childs.dataset_generator import generate_data
from childs_12.network_model_keras import KerasNetwork_12
from childs_12.predict_keras import myPredict_12

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#prepare data
samples_1 = np.load('samples_1.npy')
samples_1_train = samples_1[:-1000]#ilk 4000 line
samples_1_valid = samples_1[-1000:]#sonraki 1000 line

x_1 = samples_1_train[:,0]
y_1 = samples_1_train[:,1]

x_1_valid = samples_1_valid[:,0]
y_1_valid = samples_1_valid[:,1]
y_1_valid_onehot = to_categorical(y_1_valid, num_classes=4)

y_onehot_1 = to_categorical(y_1, num_classes=4) #one-hot encoding, y train(labels)

#optimizers
optimizer1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
optimizer2 = SGD(lr=0.1, momentum=0.1)

input_dim_1 = 1
output_dim_1 = 4

mynet_1 = KerasNetwork_1(input_dim_1, output_dim_1)

#model compile
mynet_1.model.compile(loss = "categorical_crossentropy", optimizer = optimizer1, metrics=["accuracy"])

#train
history_1 = mynet_1.model.fit(x = x_1, y = y_onehot_1, batch_size = 64, epochs = 300, validation_data = (x_1_valid,y_1_valid_onehot), verbose = 1)

print("training_1 completed")


##############################################################################################################
#prepare data
samples_12 = np.load('samples_12.npy')
samples_12_train = samples_12[:-1000]#ilk 4000 line
samples_12_valid = samples_12[-1000:]#sonraki 1000 line

x_12 = samples_12_train[:,:-1]
y_12 = samples_12_train[:,-1]

x_12_valid = samples_12_valid[:,:-1]
y_12_valid = samples_12_valid[:,-1]
y_12_valid_onehot = to_categorical(y_12_valid, num_classes=4)

y_onehot_12 = to_categorical(y_12, num_classes=4) #one-hot encoding, y train(labels)

input_dim_12 = 12
output_dim_12 = 4

mynet_12 = KerasNetwork_12(input_dim_12, output_dim_12)

#model compile
optimizer1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
mynet_12.model.compile(loss = "categorical_crossentropy", optimizer = optimizer1, metrics=["accuracy"])

#train
history_12 = mynet_12.model.fit(x = x_12, y = y_onehot_12, batch_size = 64, epochs = 300, validation_data = (x_12_valid,y_12_valid_onehot), verbose = 1)

print("training_12 completed")

#logs
plt.plot(history_1.history['loss'], color='b', label="loss(1node)")
plt.plot(history_1.history['val_loss'], color='c', label="val_loss(1node)")
plt.plot(history_12.history['loss'], color='r', label="loss(12node)")
plt.plot(history_12.history['val_loss'], color='m', label="val_loss(12node)")
plt.title("CE loss vs. epochs")
plt.xlabel("epochs")
plt.ylabel("CE loss")
plt.legend()
plt.show()

plt.plot(history_1.history['acc'], color='b', label="accuracy(1node)")
plt.plot(history_1.history['val_acc'], color='c', label="val_accuracy(1node)")
plt.plot(history_12.history['acc'], color='r', label="accuracy(12node)")
plt.plot(history_12.history['val_acc'], color='m', label="val_accuracy(12node)")
plt.title("accuracy vs. epochs")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

#prediction
x_test_1 = np.array([160])# test data-> angle:160, expected output node: 2 (chaff)
myPredict_1(x_test_1, mynet_1.model)

#-180  -150  -120  -90  -60  -30  0  30  60  90  120  150  180 :angle intervals
#     0     1     2    3    4    5  6   7   8   9   10   11    : corresponding input nodes
x_test_12 = np.array([0, 0, 0, 0, 0, -20/30, 0, 0, 0, 0, 0, 0]) #test data-> angle:-20 interval:-30 0  input node:5, expected output node:3(missile)
myPredict_12(x_test_12, mynet_12.model)





