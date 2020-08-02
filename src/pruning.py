import tensorflow as tf
import numpy as np
from typing import Callable


def magnitude_saliency_criterion(w: np.array) -> np.array:
    """Pruning criterion. The higher the absolute value of a weight, the higher its
    saliency.

    Args:
        w (numpy.array): 1-dimensional numpy array containing the weights to be pruned.

    Returns:
        np.array: vector of saliencies.
    """
    return np.abs(w)


def prune(
    model: tf.keras.Model,
    prune_proportion: float,
    criterion: Callable = magnitude_saliency_criterion,
) -> tf.keras.Model:
    """Helper function to prune neural networks defined in Keras using a specified
    criterion.

    Args:
        model (tf.keras.Model): Differentiable model defined in Keras and initialized.
        prune_proportion (float): Proportion of weights to be pruned.
        criterion (callable, optional): Criterion to prune the weights. For now it is
        defined as a function that takes a vector of weights as a parameter.
        Defaults to magnitude_saliency_criterion.

    Returns:
        tf.keras.Model: Model with new weights. There is no need to use this output as
        the model is updated by reference.
    """
    weights = {w.name: w for w in model.variables}

    # Filter out bias and mask terms
    w_mat_names = map(lambda x: x.name, model.variables)

    # Filter out bias
    prunable_w_mat_names = filter(lambda x: "_b" not in x, w_mat_names)

    # Filter out masks
    prunable_w_mat_names = filter(lambda x: "_m" not in x, prunable_w_mat_names)

    for w_name in prunable_w_mat_names:
        # Get the weights of the layer and the corresponding mask
        w = weights[w_name].numpy()
        m = weights[w_name + "_m:0"].numpy()

        # Store the original matrix shape
        shape = w.shape

        # Reshape the matrices into vectors
        w = w.reshape(-1)
        m = m.reshape(-1)

        # Calculate the number of pruned weights
        n_pruned_weights = np.round(w.size * prune_proportion).astype(int)

        # Apply the saliency criterion and sort in increasing order.
        # Get the indices of the less salient weights
        connections_to_prune = criterion(w).argsort()[:n_pruned_weights]

        # Set the weights to prune to zero (not necessary)
        w[connections_to_prune] = 0

        # Set the mask values corresponding to the pruned weights to zero
        # In order to prevent the gradient to back-propagate. Equivalent
        # to freeze individual connections
        m[connections_to_prune] = 0

        # Set the weights back to the network
        weights[w_name].assign(w.reshape(shape))
        weights[w_name + "_m:0"].assign(m.reshape(shape))

    # Important to recompile the model to get rid of the optimizer state
    model.compile(
        loss=model.loss,
        optimizer=model.optimizer._name,
        metrics=[m for m in model.metrics_names if m != "loss"],
    )
    # The weights are set by reference, there is no need to return
    # the model. We return it for potential further compatibility reasons
    return model


def restore_initial_weights(
    model: tf.keras.Model, initial_weights: dict
) -> tf.keras.Model:
    """Helper function to reinitialize the model to its initial weight values, while
    keeping the masking configuration

    Args:
        model (tf.keras.Model): Differentiable model defined in Keras and initialized.
        initial_weights (dict): dictionary of initial weights of the network. The
        structure of this dictionary must be
            key -> name of the parameter tensor in the tensorflow graph.
            value -> parameter matrix in numpy.

            The following snipped may be helpful:
            `initial_weights_backup = {w.name:w.numpy() for w in model.variables}`

    Returns:
        tf.keras.Model: Model with new weights. There is no need to use this output as
        the model is updated by reference.
    """
    # Get current weights
    weights = {w.name: w for w in model.variables}

    # Filter out bias
    initial_weights = filter(lambda x: "_b" not in x[0], initial_weights.items())

    # Filter out masks
    initial_weights = filter(lambda x: "_m" not in x[0], initial_weights)

    # Get final dict
    initial_weights = dict(initial_weights)

    for name in initial_weights:
        w = initial_weights[name]
        m = weights[name + "_m:0"]
        # Set the masked weights to zero. Not necessary, but cleaner
        w = tf.multiply(w, m)
        weights[name].assign(w)

    # Important to recompile the model to get rid of the optimizer state
    model.compile(
        loss=model.loss,
        optimizer=model.optimizer._name,
        metrics=[m for m in model.metrics_names if m != "loss"],
    )

    # The weights are set by reference, there is no need to return
    # the model. We return it for potential further compatibility reasons
    return model
