#model class
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU

class KerasNetwork:
    def __init__(self, input_dim, output_dim):

        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape = (input_dim,)))

        self.model.add(Dense(16))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(16))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.25))

        self.model.add(Dense(8))
        self.model.add(BatchNormalization())
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.1))

        self.model.add(Dense(output_dim, activation = "softmax"))



