import sys
import traceback
import pdb

import terpret_optimized_np_runtime as np_tpt
import numpy as np
import tensorflow as tf

def softmax(m, scope=""):
    with tf.name_scope(scope):
        m_rank = m.get_shape().ndims
        assert_rank_1_or_2(m_rank)
        if m_rank == 1:
            return tf.squeeze(tf.nn.softmax(tf.expand_dims(m,0)))
        else:
            return tf.nn.softmax(m)

def one_hot(indices, depth, scope=""):
    with tf.name_scope(scope):
        return tf.one_hot(indices, depth)

def make_tensor(f, input_sizes, output_size):
    tensor, output_size, error_symbol = np_tpt.make_tensor(f, input_sizes, output_size)
    return (tf.constant(tensor.astype(np.int32)), output_size, error_symbol)

def slice_out_int_literals(tensor, args):
    """
    Allow passing int literals to factors in place of marginal distributions.
    For those args which are int, specialise the tensor to handle these integer
    values & eliminate these args from the arg list.
    """ 
    n_args = len(args)
    tensor, output_size, error_symbol = tensor

    slice_begin = np.zeros((n_args)).astype(np.int32)
    slice_size = -1 * np.ones((n_args)).astype(np.int32)
    tmp_args = []
    for i,arg in enumerate(args):
        if isinstance(arg, int):
            slice_begin[i] = arg
            slice_size[i] = 1
        else:
            tmp_args.append(args[i])
    tensor = tf.squeeze(tf.slice(tensor, slice_begin, slice_size))
    args = tmp_args
    return (tensor, output_size, error_symbol), args

def assert_rank_1_or_2(arg_rank):
    assert arg_rank < 3, \
        "found argument with rank %s. " % arg_rank + \
        "Arguments should have rank 1 or 2." 

def make_batch_consistent(args, set_batch_size=None):
    """
    args[i] should be either [arg_dim] or [batch_size x arg_dim]
    if rank(args[i]) == 1 then tile to [batch_size x arg_dim]
    """
    if set_batch_size is None:
        # infer the batch_size from arg shapes
        batched_args = filter(lambda x : x.get_shape().ndims > 1, args)
        #batched_args = filter(lambda x : x.get_shape()[0].value is None, args)
        if len(batched_args) == 0:
            batch_size = 1
            is_batched = False
        else:
            # TODO: tf.assert_equal() to check that all batch sizes are consistent?
            batch_size = tf.shape(batched_args[0])[0]
            is_batched = True
    else: 
        batch_size = set_batch_size
        is_batched = True

    # tile any rank-1 args to a consistent batch_size
    tmp_args = []
    for arg in args:
        arg_rank = arg.get_shape().ndims
        assert_rank_1_or_2(arg_rank)
        if arg_rank == 1:
            tmp_args.append(tf.tile(tf.expand_dims(arg,0), [batch_size,1]))
        else:
            tmp_args.append(arg)
    args = tmp_args
    return args, is_batched


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
                for j in xrange(n_args):
                    if j == i: continue
                    args[i] = tf.expand_dims(args[i], j + 1)

            # compute joint before tensor is applied
            joint = 1
            for arg in args:
                joint = joint * arg
            
            # prepare for unsorted_segment_sum
            joint = tf.reshape(joint, (-1, np.prod(tensor_dim_sizes)))
            joint = tf.transpose(joint, [1, 0])	 # |tensor| x batch_size

            if error_symbol is not None:
                result = tf.unsorted_segment_sum(joint, tf.reshape(tensor, [-1]), output_size + 1)
                # assume error bin is last bin
                result = result[:output_size, :]
            else:
                result = tf.unsorted_segment_sum(joint, tf.reshape(tensor, [-1]), output_size)

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

        components *= weights    
        w_sum = tf.reduce_sum(components, reduction_indices=[0]) # [b x c]
        if not is_batched: w_sum = tf.squeeze(w_sum) # [c]
    return w_sum


def observe(node, observed_value, scope=""):
    with tf.name_scope(scope):
        tmp, is_batched = make_batch_consistent([node, observed_value])
        node = tmp[0]
        # numerically unstable
        SMALL_NUMBER = 1e-10
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.log(node + SMALL_NUMBER), observed_value) - SMALL_NUMBER


def normalize(value):
    return value


def test():
    sess = tf.InteractiveSession()

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
    np.testing.assert_allclose(apply_factor(z).eval(), np.array([0,1,0]))

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
    np.testing.assert_allclose(mu_z0, np.array([[0, .5, .5],
                                                [0, 1, 0],
                                                [1, 0, 0]]))

    print " mu_x + mu_y:\n", mu_z1
    np.testing.assert_allclose(mu_z1, np.array([[.25, .5, .25],
                                                [.5, .5, 0],
                                                [1, 0, 0]]))

    print " mu_x + 1:\n", mu_z2
    np.testing.assert_allclose(mu_z2, np.array([[0, .5, .5],
                                                [0, 1, 0],
                                                [1, 0, 0]]))
    
    print " 1 + mu_y:\n", mu_z3
    np.testing.assert_allclose(mu_z3, np.array([[0, .5, .5],
                                                [0, .5, .5],
                                                [0, 0, 1]]))
                                                
    print ".5 (z0 + z1):\n", mu_z
    np.testing.assert_allclose(mu_z, np.array([[.125, .5, .375],
                                               [.5, .5, 0],
                                               [1, 0, 0]]))

    print " (mu_x[0] + mu_y):\n", mu_zbatch0
    np.testing.assert_allclose(mu_zbatch0, np.array([[.25, .5, .25],
                                                    [.25, .5, .25],
                                                    [0, .5, .5]]))

    print " (mu_x[0] + mu_y[0]):\n", mu_zbatch1
    np.testing.assert_allclose(mu_zbatch1, np.array([.25, .5, .25]))

    print " .5 (mu_x + mu_y):\n", mu_zbatch2
    np.testing.assert_allclose(mu_zbatch2, np.array([[.5, .5, 0],
                                                     [.75, .25, 0],
                                                     [0, .5, .5]]))

    print " .5 (mu_x[0] + mu_y):\n", mu_zbatch3
    np.testing.assert_allclose(mu_zbatch3, np.array([[.5, .5, 0],
                                                     [.5, .5, 0],
                                                     [0.25, .75, 0]]))

    print " .5 (mu_x[0] + mu_y[0]):\n", mu_zbatch4
    np.testing.assert_allclose(mu_zbatch4, np.array([.5, .5, 0]))

    print " [.5 ...] (mu_x[0] + mu_y[0]):\n", mu_zbatch5
    np.testing.assert_allclose(mu_zbatch5, np.array([[.5, .5, 0],
                                                     [.5, .5, 0],
                                                     [.5, .5, 0]]))

if __name__ == "__main__":
    try:
        test()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
