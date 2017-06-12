import sys
import traceback
import pdb
from itertools import product

import numpy as np


"""
Numpy versions for testing. Implement using functions that are available in
tensorflow.

Support for batching is mostly in place. One exception is handling combinations
of batched/non-batched in apply_factor and weighted_sum.
"""


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    copying tensorflow's api to develop optimized version of TerpreT runtime.
    Note: here we assume data is 2D.
    """
    assert data.ndim == 2, "not implemented for D > 2"

    result = np.zeros((num_segments, data.shape[1]))
    np.add.at(result, segment_ids, data)
    return result


def make_tensor(f, input_sizes, output_size):
    """
    In this optimized version, we don't include the output dimension in the
    tensor, and instead store the output index for each input combination.
    We can then apply factors using unsorted_segment_sum (see apply_factor).
    """
    tensor = np.zeros(input_sizes, dtype=np.int)
    error_symbol = None
    for in_idx in product(*[np.arange(size) for size in input_sizes]):
        out = f(*in_idx)
        if (out < 0) or (out >= output_size):
            error_symbol = output_size
            tensor[in_idx] = error_symbol
            print "\nWarning: %i is outside the expected range. Inserting error symbol into tensor." % out
        else:
            tensor[in_idx] = f(*in_idx)
    return (tensor, output_size, error_symbol)


def apply_factor(tensor, *args):
    """
    This implementation removes the output dimension from the tensor and thus is
    O(output_size) more efficient than the unoptimized version.
    """
    tensor, output_size = tensor
    n_args = len(args)
    batch_size = args[0].shape[0]
    reshaped_args = []

    # Each arg is batch size x arg dim. Add dimensions to enable broadcasting.
    for i, arg in enumerate(args):
        shape_i = np.ones(n_args + 1, dtype=np.int)
        shape_i[0] = batch_size
        shape_i[i + 1] = -1
        reshaped_args.append(arg.reshape(*shape_i))

    # compute joint before tensor is applied
    joint = 1
    for arg in reshaped_args:
        joint = joint * arg

    # prepare for unsorted_segment_sum
    joint = joint.reshape(batch_size, -1)
    joint = joint.transpose([1, 0])  # |tensor| x batch_size

    result = unsorted_segment_sum(joint, tensor.reshape(-1), output_size)
    result = result.transpose([1, 0])
    return result


def weighted_sum(components, weights):
    """
    weights: B x M array.
    components: list of length M, with elements that are B x M' arrays
    result: B x M' array that is weighted average of components according to weights

    TODO: support case where weights and/or components aren't all batched.
    First priority: when weights is M dimensional array, which arises often.
    """
    components = np.array(components)  # M x B x M'

    # if we had einsum, this is easy...
    # return np.einsum("mbn,bm->bn", components, weights)

    # ...but we can also do with a bit of reshaping & broadcasting
    weights = weights.transpose([1, 0])  # mb
    components *= weights[:, :, np.newaxis]
    return np.sum(components, axis=0)


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
    print "    mu_x + 1:\n", mu_z0
    np.testing.assert_allclose(mu_z0, np.array([[0, .5, .5],
                                                [0, 1, 0],
                                                [1, 0, 0]]))

    mu_z1 = apply_factor(add_tensor, mu_x, mu_y)
    print " mu_x + mu_y:\n", mu_z1
    np.testing.assert_allclose(mu_z1, np.array([[.25, .5, .25],
                                                [.5, .5, 0],
                                                [1, 0, 0]]))

    mu_z = weighted_sum([mu_z0, mu_z1], mu_s)

    print ".5 (z0 + z1):\n", mu_z
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
