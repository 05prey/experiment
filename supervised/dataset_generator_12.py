#generate random data wrt rules
from random import randint
import numpy as np

def generate_data(sample_number):
    #actions
    left = 0
    right = 1
    chaff = 2
    missile = 3

    x = np.array([[0,0,0,0,0,0,15/30,0,0,0,0,0]])
    y = np.array([[3]])

    for _ in range(sample_number):

        relative_bearing = randint(-180,180)
        #rb = np.array([[relative_bearing]])

        #x = np.append(x, rb, axis=0)

        if (-180 <= relative_bearing <= -150):
            new_x = np.array([[(relative_bearing+150)/30,0,0,0,0,0,0,0,0,0,0,0]])
            action = 2

        if (-150 < relative_bearing <= -120):
            new_x = np.array([[0,(relative_bearing+120)/30,0,0,0,0,0,0,0,0,0,0]])
            action = 0

        if (-120 < relative_bearing <= -90):
            new_x = np.array([[0,0,(relative_bearing+90)/30,0,0,0,0,0,0,0,0,0]])
            action = 0

        if (-90 < relative_bearing <= -60):
            new_x = np.array([[0,0,0,(relative_bearing+60)/30,0,0,0,0,0,0,0,0]])
            action = 0

        if (-60 < relative_bearing <= -30):
            new_x = np.array([[0,0,0,0,(relative_bearing+30)/30,0,0,0,0,0,0,0]])
            action = 0

        if (-30 < relative_bearing <= 0):
            new_x = np.array([[0,0,0,0,0,(relative_bearing+0)/30,0,0,0,0,0,0]])
            action = 3

        if (0 < relative_bearing <= 30):
            new_x = np.array([[0,0,0,0,0,0,(relative_bearing-0)/30,0,0,0,0,0]])
            action = 3

        if (30 < relative_bearing <= 60):
            new_x = np.array([[0,0,0,0,0,0,0,(relative_bearing-30)/30,0,0,0,0]])
            action = 1

        if (60 < relative_bearing <= 90):
            new_x = np.array([[0,0,0,0,0,0,0,0,(relative_bearing-60)/30,0,0,0]])
            action = 1

        if (90 < relative_bearing <= 120):
            new_x = np.array([[0,0,0,0,0,0,0,0,0,(relative_bearing-90)/30,0,0]])
            action = 1

        if (120 < relative_bearing <= 150):
            new_x = np.array([[0,0,0,0,0,0,0,0,0,0,(relative_bearing-120)/30,0]])
            action = 1

        if (150 < relative_bearing <= 180):
            new_x = np.array([[0,0,0,0,0,0,0,0,0,0,0,(relative_bearing-150)/30]])
            action = 2

        x = np.append(x, new_x, axis=0)

        action = np.array([[action]])
        y = np.append(y, action, axis=0)

    samples = np.concatenate((x, y), axis=1)
    np.save('samples_12.npy', samples)

    return samples

sample_number = 5000
samples = generate_data(sample_number)

#print(np.load("x.npy"))
#print(np.load("y.npy"))
