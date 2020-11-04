#from mymodels.dataset_generator import generate_data
#from mymodels.network_model import Network

import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def generate_data(sample_number):
    #actions
    left = 0
    right = 1
    chaff = 2
    missile = 3

    x = np.array([[0]])
    y = np.array([[3]])

    for _ in range(sample_number):

        relative_bearing = randint(-180,180)
        rb = np.array([[relative_bearing]])

        x = np.append(x, rb, axis=0)

        if (-10 <= relative_bearing <= 0) or ( 0 <= relative_bearing <= 10):

            action = 3

        elif (-180 <= relative_bearing <= -170) or (170 <= relative_bearing <= 180):

            action = 2

        elif 10 < relative_bearing < 170:

            action = 1

        elif -170 < relative_bearing < -10:

            action = 0

        action = np.array([[action]])
        y = np.append(y, action, axis=0)


    #np.save("y",np.array(y))
    #np.save("x",np.array(x))

    #x = np.asarray(x)
    #y = np.asarray(y)
    sample = np.concatenate((x, y), axis=1)

    return sample

samples = generate_data(10)
x = samples[:,0]
y = samples[:,1]
#making data frame from csv file
#dataframe = pd.read_csv("data/warcraft_dataset-100.csv")

#data = np.asarray(dataframe) #dataframe to np array

#x = np.delete(data, [3], axis=1) #x train
#normalize x train
#for i in range(x.shape[0]):
#    x[i][0] = (x[i][0]-x_1min)//(x_1max-x_1min)
#    x[i][1] = (x[i][1]-x_2min)//(x_2max-x_2min)
#    x[i][2] = (x[i][2]-x_3min)//(x_3max-x_3min)

#y = np.delete(data, [0,1,2], axis=1) #y train(labels)
y = to_categorical(y, num_classes=4) #one-hot encoding, y train(labels)

# Define the optimizer
optimizer1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
optimizer2 = SGD(lr=0.1, momentum=0.1)

model = Sequential()
model.add(BatchNormalization(input_shape = (1,)))
model.add(Dense(128))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.25))
model.add(Dense(8))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(4, activation = "softmax"))
# Compile the model
model.compile(loss = "categorical_crossentropy", optimizer = optimizer1, metrics=["accuracy"])
# fit the model
history = model.fit(x = x, y = y, batch_size = 128, epochs = 300, verbose = 1)


#Plot the loss and epoch curve
plt.plot(history.history['loss'], color='b', label="loss")
plt.title("Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Predict the class
#x_test = np.array([150, 6, 40])
#x_test = x_test.reshape(-1,3)
#print("\n")
#print("Expected class: Howitzer")

#y_ = model.predict(x_test)
#print("prediction values:",y_)
# Convert predictions to class index
#y_binary_classes = np.argmax(y_,axis = 1)
#if y_binary_classes == 0:
#    print("Predicted class: Howitzer detected!")
#elif y_binary_classes == 1:
#    print("Predicted class: Cannon detected!")
# observe the weigts
#i=0
#for layer in model.layers:
#    weights = layer.get_weights()	# list of numpy arrays
#    i += 1
#    if i == 14:
#        print("\n")
#        print("final hidden layer weights:")
#        print(weights)
#print(i)

"""
RESULTS
Adam > SGD
BatchNormalization really allowed SGD optimizer to work well and operate with high learning rate(faster). SGD needs BN.
Droput improved a little bit but not so much, compatible with relu.
Crossentropy is better than MSE. Even so MSE generalized better. 
LeakyReLU and relu worked nice but they need BN. Sigmoid is not so bad. 
Softmax is better than sigmoid at the outer layer.
High batch size fitted better but needs Dropout. Must be not too big not too small.  
Dataset normalization is needed to prevent domination of big-valued feature but it co-operates with BN and SGD. 
BN can be enough for normalization&standardization of x train.   
"""


