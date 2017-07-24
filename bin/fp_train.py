#!/usr/bin/env python
'''
Usage:
    fp_train.py [options] COMPILED_MODEL DATA [TRAIN_HYPERS]

Options:
    -h --help                Show this screen.
    --train-batch NAME       Name of the data batch with training data.
    --test-batches NAME      Names of the data batches with test data.
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
import traceback
import pdb
import trainer
from custom_train import CustomTrainer
import train
import tensorflow as tf

class FPTrainer(CustomTrainer):
    @staticmethod
    def default_train_hypers():
        return {
                "optimizer": "rmsprop",
                "num_epochs": 3000,
                "stop_below_loss": 0.005,
                "learning_rate": 0.1,
                "learning_rate_decay": .9,
                "momentum": 0.0,
                "minibatch_size": -1,
                "print_frequency": 100,
                "gradientClipC2": 1.0,
                "fGradientNoise": .01,
                "fGradientNoiseGamma": .55,
                "fEntropyBonusDecayRate": .5,
                "dirichletInitScale": 2
               }

    def __init__(self, model, model_module, model_hypers, train_hypers, data,
                 do_debug=False, make_log=False):
        super(FPTrainer, self).__init__(model, model_module, model_hypers, train_hypers, data,
                                        do_debug=do_debug, make_log=make_log)


if __name__ == "__main__":
    args = docopt(__doc__)
    test_batches = args.get('--test-batches', None)
    if test_batches is not None:
        args['--test-batch'] = test_batches.split(',')

    try:
        train.load_and_run(args, FPTrainer)
    except:
        if args.get('--debug', False):
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise
