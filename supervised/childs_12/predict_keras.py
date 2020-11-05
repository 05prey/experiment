#model class
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU

def myPredict_12(x_test, model):
    x_test = x_test.reshape(-1,12)
    y_pred = model.predict(x_test)

    print("prediction values for 12 input:",y_pred)
    print("\n")
    y_binary_classes = np.argmax(y_pred,axis = 1)

    if y_binary_classes == 0:
        print("action: turn left (west)")
    elif y_binary_classes == 1:
        print("action: turn right (east)")
    elif y_binary_classes == 2:
        print("action: release chaff")
    elif y_binary_classes == 3:
        print("action: fire missile")

    return y_pred, y_binary_classes



