from unittest import TestCase
from src.layers import Linear

import tensorflow as tf
import numpy as np


class TestLinearLayer(TestCase):
    def test_compilation_and_build(self):
        # We define a small model, if no errors are raised, test passes.
        # Build MLP model
        model = tf.keras.models.Sequential([Linear(5), Linear(5)])

        # Compile
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        # Initialize
        model.build(input_shape=(32, 8))

    def test_forward_propagation(self):
        # We define a small model and pass in a dummy input. If shape is correct,
        # no nans and no infinite values appear in the output, test passes.
        # Build MLP model
        model = tf.keras.models.Sequential([Linear(5), Linear(5)])

        # Compile
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        # Initialize
        model.build(input_shape=(32, 8))

        # Dummy input
        X = np.random.randn(32, 8)
        y_hat = model(X).numpy()

        self.assertEquals((32, 5), y_hat.shape)
        self.assertEquals(False, np.isnan(y_hat.sum()))
        self.assertEquals(True, np.isfinite(y_hat.sum()))

    def test_gradient_check(self):
        # We define a small model and pass in a dummy input, get the loss, fit, and
        # get the loss again. If the loss decreases, the test passes
        model = tf.keras.models.Sequential([Linear(5), Linear(5)])

        # Compile
        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam",
        )

        # Initialize
        model.build(input_shape=(32, 8))

        # Dummy input
        X = np.random.randn(32, 8)
        y = np.random.rand(32,)
        loss_0 = model.evaluate(X, y)
        model.fit(X, y)
        loss_1 = model.evaluate(X, y)

        self.assertLess(loss_1, loss_0)

    def test_relu_activation(self):
        # Check if the output values look like relu
        model = tf.keras.models.Sequential([Linear(5, activation=tf.nn.relu)])

        # Compile
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        # Initialize
        model.build(input_shape=(32, 8))

        # Dummy input
        X = np.random.randn(32, 8)
        y_hat = model(X).numpy().reshape(-1)

        self.assertGreaterEqual(y_hat.min(), 0)
        self.assertLess(y_hat.max(), 1000)

    def test_tanh_activation(self):
        # Check if the output values look like tanh
        model = tf.keras.models.Sequential([Linear(5, activation=tf.nn.tanh)])

        # Compile
        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam",
        )

        # Initialize
        model.build(input_shape=(32, 8))

        # Dummy input
        X = np.random.randn(32, 8)
        y_hat = model(X).numpy().reshape(-1)

        self.assertGreaterEqual(y_hat.min(), -1)
        self.assertLessEqual(y_hat.max(), 1)
        self.assertGreater(y_hat.max(), 0)
        self.assertLess(y_hat.min(), 0)

    def test_ones_initialization(self):
        # Check if the network changes the initialization method. Passes if the output
        # is correct.
        model = tf.keras.models.Sequential(
            [Linear(5, activation=tf.nn.relu, initializer=tf.ones_initializer)]
        )

        # Compile
        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam",
        )

        # Initialize
        model.build(input_shape=(32, 8))

        # Dummy input
        X = np.ones((32, 8)).astype(float)
        y_hat = model(X).numpy()

        self.assertTrue(((np.ones((32, 5)) * 8) == y_hat).all())

    def test_mask_creation(self):
        # Check if the proper masks are created and initialized to zero
        model = tf.keras.models.Sequential(
            [Linear(5, activation=tf.nn.relu), Linear(10, activation=tf.nn.relu)]
        )

        # Compile
        model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam",
        )

        # Initialize
        model.build(input_shape=(32, 8))

        # Get the mask variables
        masks = list(filter(lambda x: "_m:0" in x.name, model.variables))

        self.assertEquals(2, len(masks))

        self.assertEquals((8, 5), masks[0].shape)
        self.assertTrue((np.ones((8, 5)) == masks[0].numpy()).all())

        self.assertEquals((5, 10), masks[1].shape)
        self.assertTrue((np.ones((5, 10)) == masks[1].numpy()).all())
