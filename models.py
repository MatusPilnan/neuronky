import keras
from keras import Model
from keras.layers import Conv2D, BatchNormalization, Subtract, Activation


class DnCNN(Model):
    def __init__(self, depth):
        super().__init__()

        self.model_layers = []
        self.model_layers.append(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same', activation='relu'))
        for i in range(depth - 2):
            self.model_layers.append(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False))
            self.model_layers.append(BatchNormalization())
            self.model_layers.append(Activation('relu'))

        self.model_layers.append(Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False))
        self.model_layers.append(Subtract())

    def call(self, input, **kwargs):
        x = input
        for layer in self.model_layers[:-1]:
            x = layer(x)
        x = self.model_layers[-1]([input, x])

        return x


def dcnn_loss(predicted, true):
    return keras.backend.sum(keras.backend.square(predicted - true)) / 2
