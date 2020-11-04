#generate random data wrt rules
from random import randint
import numpy as np

def generate_data():

    _y = [ ]
    _x = [ ]

    sample =  10
    for _ in range(sample):

        relative_bearing = randint(-180,180)

        _x.append([relative_bearing])

        if (-10 <= relative_bearing <= 0) or ( 0 <= relative_bearing <= 10):

            y = [0,0,0,1]

        elif (-180 <= relative_bearing <= -170) or (170 <= relative_bearing <= 180):

            y = [0,0,1,0]

        elif 10 < relative_bearing < 170:

            y = [0,1,0,0]

        elif -170 < relative_bearing < -10:

            y = [1,0,0,0]

        _y.append(y)

    np.save("y",np.array(_y))
    np.save("x",np.array(_x))

#collect_data()

#print(np.load("x.npy"))
#print(np.load("y.npy"))
