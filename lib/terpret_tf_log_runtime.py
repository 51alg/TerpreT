import sys
import traceback
import pdb

# import terpret_optimized_np_runtime as np_tpt
import numpy as np
import tensorflow as tf

from terpret_tf_runtime import (
    make_tensor,
    slice_out_int_literals,
    assert_rank_1_or_2,
    make_batch_consistent)

SMALL_NUMBER = 1e-16

def softmax(m, scope=""):
    with tf.name_scope(scope):
        return m - logsumexp(m, reduction_indices=m.get_shape().ndims-1, keep_dims=True)

def one_hot(indices, depth, scope=""):
    with tf.name_scope(scope):
        return tf.log(tf.one_hot(indices, depth) + SMALL_NUMBER)

def handle_inf(x):
    return tf.log(tf.exp(x) + SMALL_NUMBER)

def logsumexp(v, reduction_indices=None, keep_dims=False):
    if float(tf.__version__[:4]) > 0.10: # reduce_logsumexp does not exist below tfv0.11
        if isinstance(reduction_indices, int): # due to a bug in tfv0.11
            reduction_indices = [reduction_indices]
        return handle_inf(
                 tf.reduce_logsumexp(v,
                  reduction_indices, # this is a bit fragile. reduction_indices got renamed to axis in tfv0.12
                  keep_dims=keep_dims)
                 )
    else:
        m = tf.reduce_max(v, reduction_indices=reduction_indices, keep_dims=keep_dims)
        # Use SMALL_NUMBER to handle v = []
        return m + tf.log(tf.reduce_sum(tf.exp(v - m), 
                        reduction_indices=reduction_indices,
                        keep_dims=keep_dims) + SMALL_NUMBER)

def apply_factor(tensor, *args, **kwargs):
    scope = kwargs.pop("scope", "")     
    with tf.name_scope(scope):
        n_args = len(args)

        if n_args is 0:
            tensor, output_size, error_symbol = tensor
            return one_hot(tensor, output_size, scope=scope)
        else:
            tensor, args = slice_out_int_literals(tensor, list(args))
            args, is_batched = make_batch_consistent(args)
            tensor, output_size, error_symbol = tensor

            # handle the case where all arguments were int literals
            tensor_dim_sizes = [dim.value for dim in tensor.get_shape()]
            if not tensor_dim_sizes:
                return one_hot(tensor, output_size, scope=scope)

            # Each arg is batch size x arg dim. Add dimensions to enable broadcasting.
            for i, arg in enumerate(args):
                for j in range(len(args)):
                    if j == i: continue
                    args[i] = tf.expand_dims(args[i], j + 1)

            # compute joint before tensor is applied
            joint = 0
            for arg in args:
                joint = joint + arg

            # prepare for unsorted_segment_sum
            joint = tf.reshape(joint, (-1, np.prod(tensor_dim_sizes)))
            joint = tf.transpose(joint, [1, 0])  # |tensor| x batch_size

            flat_tensor = tf.reshape(tensor, [-1])
            if error_symbol is not None:
                to_logsumexp = tf.dynamic_partition(joint, flat_tensor, output_size + 1)
                del to_logsumexp[error_symbol]
            else:
                to_logsumexp = tf.dynamic_partition(joint, flat_tensor, output_size)



            result = tf.pack(
                        map(lambda x : logsumexp(x, reduction_indices=0), to_logsumexp)
                    )

            result = tf.transpose(result, [1, 0])
            if not is_batched: result = tf.squeeze(result)
            return result

def weighted_sum(components, weights, scope=""):
    # n: num_components
    # b: batch_size
    # c: component_size
    with tf.name_scope(scope):
        weight_is_batched = (weights.get_shape().ndims == 2)
        if weight_is_batched:
            set_batch_size = tf.shape(weights)[0]
        else:
            set_batch_size = None
        components, is_batched = make_batch_consistent(components, set_batch_size=set_batch_size)
        components = tf.pack(components) # [n x b x c]
        
        weight_rank = weights.get_shape().ndims
        assert_rank_1_or_2(weight_rank)
        if weight_rank == 1:
            weights = tf.reshape(weights, [-1,1,1]) # [n x 1 x 1]
        elif weight_rank == 2:
            weights = tf.expand_dims(tf.transpose(weights, [1, 0]),2) # [n x b x 1]
            
        components += weights
        # TODO: change this to tf.reduce_logsumexp when it is relased
        w_sum = logsumexp(components, reduction_indices=0) # [b x c]
        if not is_batched: w_sum = tf.squeeze(w_sum) # [c]
    return w_sum

def observe(node, observed_value, scope=""):
    with tf.name_scope(scope):
        tmp, is_batched = make_batch_consistent([node, observed_value])
        node = tmp[0]
        return tf.nn.sparse_softmax_cross_entropy_with_logits(node, observed_value) 


def normalize(value):
    if value.ndim == 1:
        return np.exp(value) / np.sum(np.exp(value))
    elif value.ndim == 2:
        return np.exp(value) / np.sum(np.exp(value), 1)[:, None]
    else:
        # Don't even try to normalise, but shouldn't happen anyway
        return value


def test():
    SMALL_NUMBER = 1e-16

    sess = tf.InteractiveSession()

    # make some marginals
    mu_s = np.log(np.array([[.5, .5],
                     [0, 1],
                     [.75, .25]]) + SMALL_NUMBER)  # switch variable

    size = 3
    mu_x = np.log(np.array([[.5, .5, 0],
                     [1, 0, 0],
                     [0, 0, 1]]) + SMALL_NUMBER)
    mu_y = np.log(np.array([[.5, .5, 0],
                     [.5, .5, 0],
                     [0, 1, 0]]) + SMALL_NUMBER)
                     

    mu_s = tf.Variable(mu_s)
    mu_x = tf.Variable(mu_x)
    mu_y = tf.Variable(mu_y)

    for v in [mu_s, mu_x, mu_y]:
        v.initializer.run()

    # make some tensors
    def increment(x):
        return (x + 1) % size
    increment_tensor = make_tensor(increment, [size], size)

    def add(x, y):
        return (x + y) % size
    add_tensor = make_tensor(add, [size, size], size)

    def one():
        return 1
    
    z = make_tensor(one, [], 3)
    np.testing.assert_allclose(np.exp(apply_factor(z).eval()), np.array([0,1,0]),
        atol = 10*SMALL_NUMBER)

    # test constant argument to apply_factor:
    const_fact = apply_factor(increment_tensor, 0)

    # test factors
    mu_z0 = apply_factor(increment_tensor, mu_x)
    mu_z1 = apply_factor(add_tensor, mu_x, mu_y)
    # test integers as arguments
    mu_z2 = apply_factor(add_tensor, mu_x, 1)
    mu_z3 = apply_factor(add_tensor, 1, mu_y)
    # test weighted sum
    mu_z = weighted_sum([mu_z0, mu_z1], mu_s)

    # test mixed batching
    mu_zbatch0 = apply_factor(add_tensor, mu_x[0,:], mu_y)
    mu_zbatch1 = apply_factor(add_tensor, mu_x[0,:], mu_y[0,:])
    mu_zbatch2 = weighted_sum([mu_x, mu_y], mu_s[0,:])
    mu_zbatch3 = weighted_sum([mu_x[0,:], mu_y], mu_s[0,:])
    mu_zbatch4 = weighted_sum([mu_x[0,:], mu_y[0,:]], mu_s[0,:])
    mu_zbatch5 = weighted_sum([mu_x[0,:], mu_y[0,:]], mu_s)    

    # evaluate and test
    mu_z0 = mu_z0.eval()
    mu_z1 = mu_z1.eval()
    mu_z2 = mu_z2.eval()
    mu_z3 = mu_z3.eval()
    mu_z = mu_z.eval()
    mu_zbatch0 = mu_zbatch0.eval()
    mu_zbatch1 = mu_zbatch1.eval()
    mu_zbatch2 = mu_zbatch2.eval()
    mu_zbatch3 = mu_zbatch3.eval()
    mu_zbatch4 = mu_zbatch4.eval()
    mu_zbatch5 = mu_zbatch5.eval()


    print " mu_x + 1:\n", mu_z0
    np.testing.assert_allclose(np.exp(mu_z0), np.array([[0, .5, .5],
                                                [0, 1, 0],
                                                [1, 0, 0]]),
                                                atol = 10*SMALL_NUMBER)

    print " mu_x + mu_y:\n", mu_z1
    np.testing.assert_allclose(np.exp(mu_z1), np.array([[.25, .5, .25],
                                                [.5, .5, 0],
                                                [1, 0, 0]]),
                                                atol = 10*SMALL_NUMBER)

    print " mu_x + 1:\n", mu_z2
    np.testing.assert_allclose(np.exp(mu_z2), np.array([[0, .5, .5],
                                                [0, 1, 0],
                                                [1, 0, 0]]),
                                                atol = 10*SMALL_NUMBER)
    
    print " 1 + mu_y:\n", mu_z3
    np.testing.assert_allclose(np.exp(mu_z3), np.array([[0, .5, .5],
                                                [0, .5, .5],
                                                [0, 0, 1]]),
                                                atol = 10*SMALL_NUMBER)
                                                
    print ".5 (z0 + z1):\n", mu_z
    np.testing.assert_allclose(np.exp(mu_z), np.array([[.125, .5, .375],
                                               [.5, .5, 0],
                                               [1, 0, 0]]),
                                               atol = 10*SMALL_NUMBER)

    print " (mu_x[0] + mu_y):\n", mu_zbatch0
    np.testing.assert_allclose(np.exp(mu_zbatch0), np.array([[.25, .5, .25],
                                                    [.25, .5, .25],
                                                    [0, .5, .5]]),
                                                    atol = 10*SMALL_NUMBER)

    print " (mu_x[0] + mu_y[0]):\n", mu_zbatch1
    np.testing.assert_allclose(np.exp(mu_zbatch1), np.array([.25, .5, .25]),
                               atol = 10*SMALL_NUMBER)

    print " .5 (mu_x + mu_y):\n", mu_zbatch2
    np.testing.assert_allclose(np.exp(mu_zbatch2), np.array([[.5, .5, 0],
                                                     [.75, .25, 0],
                                                     [0, .5, .5]]),
                                                     atol = 10*SMALL_NUMBER)

    print " .5 (mu_x[0] + mu_y):\n", mu_zbatch3
    np.testing.assert_allclose(np.exp(mu_zbatch3), np.array([[.5, .5, 0],
                                                     [.5, .5, 0],
                                                     [0.25, .75, 0]]),
                                                     atol = 10*SMALL_NUMBER)

    print " .5 (mu_x[0] + mu_y[0]):\n", mu_zbatch4
    np.testing.assert_allclose(np.exp(mu_zbatch4), np.array([.5, .5, 0]),
                               atol = 10*SMALL_NUMBER)

    print " [.5 ...] (mu_x[0] + mu_y[0]):\n", mu_zbatch5
    np.testing.assert_allclose(np.exp(mu_zbatch5), np.array([[.5, .5, 0],
                                                     [.5, .5, 0],
                                                     [.5, .5, 0]]),
                                                     atol = 10*SMALL_NUMBER)
                           

if __name__ == "__main__":
    try:
        test()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
