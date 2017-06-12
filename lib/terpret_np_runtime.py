import sys
import traceback
import pdb
from itertools import product

import numpy as np


"""
Numpy versions for testing. Don't worry too much about maximum efficiency,
although looking forward, we might want to use these at test time so don't
make it too inefficient.

Support for batching is in place.
"""


def make_tensor(f, input_sizes, output_size):
    tensor = np.zeros(input_sizes + [output_size])
    for in_idx in product(*[np.arange(size) for size in input_sizes]):
        out_idx = f(*in_idx)
        idx = in_idx + (out_idx,)
        tensor[idx] = 1
    return tensor


def apply_factor(tensor, *args):
    n_args = len(args)
    batch_size = args[0].shape[0]
    reshaped_args = []
    args += (np.ones((batch_size, tensor.shape[-1])),)  # for output dimension

    # each arg is batch size x arg dim
    for i, arg in enumerate(args):
        shape_i = np.ones(n_args + 2, dtype=np.int)
        shape_i[0] = batch_size
        shape_i[i + 1] = -1
        reshaped_args.append(arg.reshape(*shape_i))

    joint = 1
    for arg in reshaped_args:
        joint = joint * arg
    joint *= tensor[np.newaxis, ...]
    result = np.sum(joint, axis=tuple(1 + np.arange(n_args)))
    return result


def weighted_sum(components, weights):
    """
    weights: B x M array. TODO: support case where it is M dimensional array
    components: list of length M, with elements that are B x M' arrays
    result: B x M' array that is weighted average of components according to weights
    """

    result = 0
    for i in xrange(weights.shape[1]):
        print "weight %s:\n" % i, weights[:, [i]]
        print "component %s:\n" % i, components[i]
        result += weights[:, [i]] * components[i]
    return result


def test():
    # make some marginals
    mu_s = np.array([[.5, .5],
                     [0, 1],
                     [.75, .25]])  # switch variable

    size = 3
    mu_x = np.array([[.5, .5, 0],
                     [1, 0, 0],
                     [0, 0, 1]])
    mu_y = np.array([[.5, .5, 0],
                     [.5, .5, 0],
                     [0, 1, 0]])

    # make some tensors
    def increment(x):
        return (x + 1) % size
    increment_tensor = make_tensor(increment, [size], size)

    def add(x, y):
        return (x + y) % size
    add_tensor = make_tensor(add, [size, size], size)

    # apply factors
    mu_z0 = apply_factor(increment_tensor, mu_x)
    print "    mu_x + 1:\t", mu_z0
    np.testing.assert_allclose(mu_z0, np.array([[0, .5, .5],
                                                [0, 1, 0],
                                                [1, 0, 0]]))

    mu_z1 = apply_factor(add_tensor, mu_x, mu_y)
    print " mu_x + mu_y:\t", mu_z1
    np.testing.assert_allclose(mu_z1, np.array([[.25, .5, .25],
                                                [.5, .5, 0],
                                                [1, 0, 0]]))

    mu_z = weighted_sum([mu_z0, mu_z1], mu_s)

    print ".5 (z0 + z1):\t", mu_z
    np.testing.assert_allclose(mu_z, np.array([[.125, .5, .375],
                                               [.5, .5, 0],
                                               [1, 0, 0]]))


if __name__ == "__main__":
    try:
        test()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
