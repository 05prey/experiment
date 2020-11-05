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
        rb = np.array([[relative_bearing / 180]])

        x = np.append(x, rb, axis=0)

        if (-30 <= relative_bearing <= 0) or ( 0 <= relative_bearing <= 30):

            action = 3

        elif (-180 <= relative_bearing <= -150) or (150 <= relative_bearing <= 180):

            action = 2

        elif 30 < relative_bearing < 150:

            action = 1

        elif -150 < relative_bearing < -30:

            action = 0

        action = np.array([[action]])
        y = np.append(y, action, axis=0)

    samples = np.concatenate((x, y), axis=1)
    np.save('samples_1.npy', samples)

    return samples

sample_number = 5000
samples = generate_data(sample_number)

#print(np.load("x.npy"))
#print(np.load("y.npy"))
