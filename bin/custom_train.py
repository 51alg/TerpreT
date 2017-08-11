#!/usr/bin/env python
'''
Usage:
    custom_train.py [options] COMPILED_MODEL DATA [TRAIN_HYPERS]

Options:
    -h --help                Show this screen.
    --train-batch NAME       Name of the data batch with training data.
    --test-batch NAME        Name of the data batch with test data.
    --validation-batch NAME  Name of the data batch with validation data.
    --store-data FILE        Store training result as HDF5 file in FILE.
    --print-params           Display parameters on CLI after training.
    --print-loss-breakdown   Display contribution of individual loss components
                             after final iteration.
    --override-hypers HYPERS Overrides for hyperparameters.
    --debug                  Enable debug routines. [default: False]
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from docopt import docopt
import sys
import pdb
import traceback
import numpy as np
import tensorflow as tf
from trainer import Trainer
import train
from tensorflow.python.ops import clip_ops

class CustomTrainer(Trainer):
    """
    This class demonstrates how to override Trainer so as to implement custom
    optimization techniques.
    """

    GRADIENT_CLIP        = "gradientClipC2"
    GRADIENT_NOISE       = "fGradientNoise"
    GRADIENT_NOISE_DECAY = "fGradientNoiseGamma"
    ENTROPY              = "fEntropyBonus"
    ENTROPY_DECAY        = "fEntropyBonusDecayRate"
    INIT_SCALE           = "dirichletInitScale"

    @staticmethod
    def default_train_hypers():
        return {
            "optimizer": "rmsprop",
            "num_epochs": 1000,
            "stop_below_loss": 0.00001,
            "learning_rate": 0.1,
            "learning_rate_decay": .9,
            "momentum": 0,
            "minibatch_size": -1,
            "print_frequency": 10,
            "gradientClipC2": 1.0,
            "fGradientNoise": .3,
            "fGradientNoiseGamma": .55,
            "fEntropyBonus": 1,
            "fEntropyBonusDecayRate": 0.005, # 1 / number of iterations for entropy bonus to decay by factor e^-1
            "dirichletInitScale": 1
        }

    def __init__(self, model, model_module, model_hypers, train_hypers, data,
                 seed=0, do_debug=False, make_log=False):
        super(CustomTrainer, self).__init__(model, model_module, model_hypers, train_hypers, data,
                                        seed=seed, do_debug=do_debug, make_log=make_log)

    def param_init_function(self, size):
        return self.log_dirichlet(size, self.train_hypers["dirichletInitScale"])

    def apply_update(self, optimizer, grads_and_vars):
        (grads, vars) = zip(*grads_and_vars)
        
        # Gradient clipping
        if CustomTrainer.GRADIENT_CLIP in self.train_hypers:
            grads, global_norm = clip_ops.clip_by_global_norm(grads,
                                    self.train_hypers[CustomTrainer.GRADIENT_CLIP])
        # Gradient noise
        if CustomTrainer.GRADIENT_NOISE in self.train_hypers:
            sigma_sqr = self.train_hypers[CustomTrainer.GRADIENT_NOISE]
            if CustomTrainer.GRADIENT_NOISE_DECAY in self.train_hypers:
                sigma_sqr /= tf.pow(1.0 + tf.to_float(self.global_step),
                                    self.train_hypers[CustomTrainer.GRADIENT_NOISE_DECAY])
            grads_tmp = []
            for g in grads:
                if g is not None:
                    noisy_grad = g + tf.sqrt(sigma_sqr)*tf.random_normal(tf.shape(g))
                    grads_tmp.append(noisy_grad)
                else:
                    grads_tmp.append(g)
            grads = grads_tmp
            
        train_op = optimizer.apply_gradients(zip(grads, vars), global_step=self.global_step)
        return train_op

    def get_entropy(self):
        log_params = self.model.params.values()
        entropies = []
        for m in log_params:
            assert m.get_shape().ndims == 1, "Params should have rank 1. " + \
                "Found parameter with rank %s" % m.get_shape().ndims 
            max_m = tf.reduce_max(m)
            entropies.append(-tf.reduce_sum(m * tf.exp(m-max_m))*tf.exp(max_m))

        if len(entropies) > 0:
            total_entropy = tf.add_n(entropies)
        else:
            total_entropy = 0
        return total_entropy           

    def construct_loss(self, output_nodes):
        loss, display_loss, output_datas, output_masks, loss_nodes = super(CustomTrainer, self).construct_loss(output_nodes)

        # Entropy
        if CustomTrainer.ENTROPY in self.train_hypers:
            entropy_bonus = self.train_hypers[CustomTrainer.ENTROPY]
            entropy = self.get_entropy()
            if CustomTrainer.ENTROPY_DECAY in self.train_hypers:
                entropy_bonus = tf.exp(-tf.to_float(self.global_step) * 
                    self.train_hypers[CustomTrainer.ENTROPY_DECAY])
            weighted_entropy = entropy_bonus * entropy
            display_loss = loss
            loss = loss - weighted_entropy
        
        return loss, display_loss, output_datas, output_masks, loss_nodes

    def param_init_function(self, size):
        if CustomTrainer.INIT_SCALE in self.train_hypers:
            s = self.train_hypers[CustomTrainer.INIT_SCALE]
        else: s = 1
        return np.log(np.random.dirichlet(s * np.ones(size)).astype(np.float32))     

if __name__ == "__main__":
    args = docopt(__doc__)

    try:
        train.load_and_run(args, CustomTrainer)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
