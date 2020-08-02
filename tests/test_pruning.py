from unittest import TestCase
from src.pruning import (
    prune,
    magnitude_saliency_criterion,
    random_sailency_criterion,
    restore_initial_weights,
)
from src.layers import Linear

import tensorflow as tf
import numpy as np


class TestPruneFunction(TestCase):
    def test_pruning_amount(self):
        # Check masks and weight matrices. Passes if the amount being clamped to zero is
        # very close to the desired amount
        # Build MLP model
        model = tf.keras.models.Sequential(
            [Linear(128), Linear(128), Linear(10, activation=tf.nn.softmax)]
        )

        # Compile
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        # Initialize
        model.build(input_shape=(128, 765))

        prune(model=model, prune_proportion=0.75, criterion=random_sailency_criterion)

        for w in model.variables:
            if "_b:0" not in w.name:
                self.assertAlmostEquals(0.75, (w.numpy() == 0).mean(), 3)

    def test_pruning_freeze(self):
        # We train, prune, and train again. The test passes if the pruning holds
        # after the second fitting.
        # Build MLP model
        model = tf.keras.models.Sequential(
            [Linear(128), Linear(128), Linear(10, activation=tf.nn.softmax)]
        )

        # Compile
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        # Initialize
        model.build(input_shape=(128, 50))

        # Generate dummy data
        X = np.random.randn(10000, 50)
        y = np.random.choice(range(10), size=10000)

        model.fit(X, y)

        model = prune(
            model=model, prune_proportion=0.75, criterion=random_sailency_criterion
        )

        for w in model.variables:
            if "_b:0" not in w.name:
                self.assertAlmostEquals(0.75, (w.numpy() == 0).mean(), 3)

        model.fit(X, y)

        # Here the proportions should not increase! If they do, revise the freezing
        # process. Also check the optimizer parameters, if the model is not recompiled
        # those params make the weights move a bit.
        for w in model.variables:
            if "_b:0" not in w.name:
                self.assertAlmostEquals(0.75, (w.numpy() == 0).mean(), 3)

    def test_pruning_consistency(self):
        # We prune once, check the proportion pruned, and then prune more times with
        # the same proportion. The test passes if the pruning does not increase
        # Build MLP model
        model = tf.keras.models.Sequential(
            [Linear(128), Linear(128), Linear(10, activation=tf.nn.softmax)]
        )

        # Compile
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        # Initialize
        model.build(input_shape=(128, 50))

        # Generate dummy data
        X = np.random.randn(10000, 50)
        y = np.random.choice(range(10), size=10000)

        model.fit(X, y)

        model = prune(
            model=model, prune_proportion=0.85, criterion=magnitude_saliency_criterion
        )

        for w in model.variables:
            if "_b:0" not in w.name:
                self.assertAlmostEquals(0.85, (w.numpy() == 0).mean(), 3)

        model = prune(
            model=model, prune_proportion=0.85, criterion=magnitude_saliency_criterion
        )
        model = prune(
            model=model, prune_proportion=0.85, criterion=magnitude_saliency_criterion
        )
        model = prune(
            model=model, prune_proportion=0.85, criterion=magnitude_saliency_criterion
        )

        # Here the proportions should not increase given that the magnitude saliency
        # criterion should re-prune the same weights
        for w in model.variables:
            if "_b:0" not in w.name:
                self.assertAlmostEquals(0.85, (w.numpy() == 0).mean(), 3)


class TestReinitializeFunction(TestCase):
    def test_reinitialization_work_pple(self):
        # We define one model, then perform a manual modification to the weights, and
        # thirdly we reinitialize the model. The test passes if the model with initial
        # and reinitialized values return the same output given the same input.

        # Build MLP model
        model = tf.keras.models.Sequential(
            [Linear(128), Linear(128), Linear(10, activation=tf.nn.softmax)]
        )

        # Compile
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        # Initialize
        model.build(input_shape=(128, 50))

        initial_weights_backup = {w.name: w.numpy() for w in model.variables}

        # Generate dummy data
        X = np.random.randn(10000, 50)

        initial_y_hat = model(X)

        model.variables[0].assign(model.variables[0].numpy() * 1.5)

        mod_y_hat = model(X)

        restore_initial_weights(model=model, initial_weights=initial_weights_backup)

        reinit_y_hat = model(X)

        self.assertGreater(np.sum(np.abs(initial_y_hat - mod_y_hat)), 0)
        self.assertEquals(np.sum(np.abs(initial_y_hat - reinit_y_hat)), 0)

    def test_freezing_consistency_with_pruning(self):
        # We train, prune, and train again. The test passes if the pruning holds
        # after the second fitting.
        # Build MLP model
        model = tf.keras.models.Sequential(
            [Linear(128), Linear(128), Linear(10, activation=tf.nn.softmax)]
        )

        # Compile
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

        # Initialize
        model.build(input_shape=(128, 50))

        initial_weights_backup = {w.name: w.numpy() for w in model.variables}

        # Generate dummy data
        X = np.random.randn(10000, 50)
        y = np.random.choice(range(10), size=10000)

        model.fit(X, y)

        model = prune(
            model=model, prune_proportion=0.75, criterion=random_sailency_criterion
        )

        model.fit(X, y)

        restore_initial_weights(model=model, initial_weights=initial_weights_backup)

        # The 0.75 proportion should be hold even after reinitialization.
        for w in model.variables:
            if "_b:0" not in w.name:
                self.assertAlmostEquals(0.75, (w.numpy() == 0).mean(), 3)

