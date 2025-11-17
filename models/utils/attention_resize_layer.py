import tensorflow as tf
from tensorflow.keras.layers import Layer

class ResizeLayer(Layer):
    def __init__(self):
        super(ResizeLayer, self).__init__()

    def call(self, inputs):
        size = (tf.shape(inputs)[1], tf.shape(inputs)[2])
        return tf.image.resize(inputs, size=size)    