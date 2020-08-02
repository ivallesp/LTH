import tensorflow as tf
import os
from typing import Callable


# Here we define a custom fully connected layer with a mask to be able to
class Linear(tf.keras.layers.Layer):
    def __init__(
        self,
        n_units: int = 32,
        activation: Callable = tf.nn.relu,
        initializer: Callable = tf.random_normal_initializer,
    ):
        """Custom Fully Connected layer with Keras compatibility in TensorFlow >=2.2.
        The current layer implements a non-trainable masking mechanism that enables
        easily freezing individual parameters.

        Args:
            n_units (int, optional): Number of neurons. Defaults to 32.
            activation (tf.nn function, optional): Activation function; it needs to be a
            differentiable function. Defaults to tf.nn.relu.
            initializer (tf initializer, optional): Tensorflow initializer. Defaults to
            tf.random_normal_initializer.
        """
        super(Linear, self).__init__()
        self.activation = activation
        self.n_units = n_units
        self.initializer = initializer

    def build(self, input_shape: tuple):
        input_dim = input_shape[-1]
        # Main matrix of weights definition
        w_init = self.initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, self.n_units), dtype="float32"),
            trainable=True,
        )

        # Mask for the main matrix of weights; unmasked by default (all to ones)
        m_init = tf.ones_initializer()
        self.w_mask = tf.Variable(
            initial_value=m_init(shape=(input_dim, self.n_units), dtype="float32"),
            trainable=False,
            name=os.path.split(self.w.name)[-1] + "_m",
        )

        # Bias vector definition
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.n_units,), dtype="float32"),
            trainable=True,
            name=os.path.split(self.w.name)[-1] + "_b",
        )

    def call(self, inputs: tf.Tensor):
        # Apply the mask to the weights
        w = tf.multiply(self.w, self.w_mask)
        # Apply the linear transformation
        out = tf.matmul(inputs, w) + self.b
        # Apply the activation function
        out = self.activation(out)
        return out
