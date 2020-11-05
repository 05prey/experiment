#generate random data wrt rules
from random import randint
import numpy as np


def generate_data(sample_number):
#actions:        left right chaff missile
#output nodes:    0     1     2     3

    x_1 = np.array([[90/180]])
    y_1 = np.array([[1]])

    x_12 = np.array([[0,0,0,0,0,0,0,0,(90-60)/30,0,0,0]])
    y_12 = np.array([[1]])

    for _ in range(sample_number):

        relative_bearing = randint(-180,180)
        rb = np.array([[relative_bearing / 180]])

        x_1 = np.append(x_1, rb, axis=0)

        if (-15 <= relative_bearing <= 0) or ( 0 <= relative_bearing <= 15):

            action = 3

        elif (-180 <= relative_bearing <= -165) or (165 <= relative_bearing <= 180):

            action = 2

        elif 15 < relative_bearing < 165:

            action = 1

        elif -165 < relative_bearing < -15:

            action = 0

        action_arr1 = np.array([[action]])
        y_1 = np.append(y_1, action_arr1, axis=0)

###############################################################################################################################################################
#intervals: -15 15     15 45     45 75     75 105     105 135     135 165     165 -165     -165 -135     -135 -105     -105 -75     -75 -45     -45 -15
#input nodes:  0         1         2         3           4           5           6             7             8             9           10          11

#intervals: -180 -150   -150 -120     -120 -90     -90 -60    -60 -30     -30 0    0 30     30 60     60 90     90 120     120 150    150 180
#input nodes:   0           1             2           3          4           5       6        7         8         9           10         11



        if (-180 <= relative_bearing <= -150):
            new_x = np.array([[(relative_bearing+150)/30,0,0,0,0,0,0,0,0,0,0,0]])


        elif (-150 < relative_bearing <= -120):
            new_x = np.array([[0,(relative_bearing+120)/30,0,0,0,0,0,0,0,0,0,0]])


        elif (-120 < relative_bearing <= -90):
            new_x = np.array([[0,0,(relative_bearing+90)/30,0,0,0,0,0,0,0,0,0]])


        elif (-90 < relative_bearing <= -60):
            new_x = np.array([[0,0,0,(relative_bearing+60)/30,0,0,0,0,0,0,0,0]])


        elif (-60 < relative_bearing <= -30):
            new_x = np.array([[0,0,0,0,(relative_bearing+30)/30,0,0,0,0,0,0,0]])
            action = 0

        elif (-30 < relative_bearing <= 0):
            new_x = np.array([[0,0,0,0,0,(relative_bearing+0)/30,0,0,0,0,0,0]])


        elif (0 < relative_bearing <= 30):
            new_x = np.array([[0,0,0,0,0,0,(relative_bearing-0)/30,0,0,0,0,0]])


        elif (30 < relative_bearing <= 60):
            new_x = np.array([[0,0,0,0,0,0,0,(relative_bearing-30)/30,0,0,0,0]])


        elif (60 < relative_bearing <= 90):
            new_x = np.array([[0,0,0,0,0,0,0,0,(relative_bearing-60)/30,0,0,0]])


        elif (90 < relative_bearing <= 120):
            new_x = np.array([[0,0,0,0,0,0,0,0,0,(relative_bearing-90)/30,0,0]])


        elif (120 < relative_bearing <= 150):
            new_x = np.array([[0,0,0,0,0,0,0,0,0,0,(relative_bearing-120)/30,0]])


        elif (150 < relative_bearing <= 180):
            new_x = np.array([[0,0,0,0,0,0,0,0,0,0,0,(relative_bearing-150)/30]])



        x_12 = np.append(x_12, new_x, axis=0)

        action_arr12 = np.array([[action]])
        y_12 = np.append(y_12, action_arr12, axis=0)


    samples_1 = np.concatenate((x_1, y_1), axis=1)
    np.save('samples_1.npy', samples_1)

    samples_12 = np.concatenate((x_12, y_12), axis=1)
    np.save('samples_12.npy', samples_12)


    return samples_1, samples_12


sample_number = 5000
samples_1, samples_12 = generate_data(sample_number)



