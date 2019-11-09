from keras import Model
import keras
from keras.layers import Layer, Conv3D, ReLU, BatchNormalization


class ConvReLU(Layer):
    def __init__(self, kernel_size=(3, 3, 3)):
        super().__init__()

        self.conv = Conv3D(filters=64, kernel_size=kernel_size)
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        return self.relu(x)


class ConvBNReLU(Layer):
    def __init__(self, kernel_size=(3, 3, 64)):
        super().__init__()

        self.conv = Conv3D(filters=64, kernel_size=kernel_size)
        self.bn = BatchNormalization()
        self.relu = ReLU()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.relu(x)


class DnCNN(Model):
    def __init__(self, depth):
        super().__init__()

        self.model_layers = []
        self.model_layers.append(ConvReLU())
        for i in range(depth - 2):
            self.model_layers.append(ConvBNReLU())
        self.model_layers.append(Conv3D(filters=3, kernel_size=(3, 3, 64)))

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x


def dcnn_loss(predicted, true):
    return keras.backend.sum(keras.backend.square(predicted - true))/2