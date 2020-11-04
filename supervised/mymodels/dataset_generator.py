#generate random data wrt rules
from random import randint
import numpy as np

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

sample = generate_data(10)

#print(np.load("x.npy"))
#print(np.load("y.npy"))
