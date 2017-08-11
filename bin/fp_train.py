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
                 seed=0, do_debug=False, make_log=False):
        super(FPTrainer, self).__init__(model, model_module, model_hypers, train_hypers, data,
                                        seed=seed, do_debug=do_debug, make_log=make_log)

    def construct_output_loss_nodes(self, output_nodes):
        output_datas = {}
        loss_nodes = {}

        # group corresponding outputListIsDone and outputListVal output vars together
        outputListIsDone_names = {}
        outputListVal_names = {}
        for var_name in output_nodes.keys():
            if var_name.startswith("outputListIsDone_"):
                i = int(var_name[len("outputListIsDone_"):])
                outputListIsDone_names[i] = var_name
            elif var_name.startswith("outputListVal_"):
                i = int(var_name[len("outputListVal_"):])
                outputListVal_names[i] = var_name

        assert set(outputListIsDone_names.keys()) == set(outputListVal_names.keys())
        name_pairs = []
        for i in outputListIsDone_names.keys():
            name_pairs.append((outputListIsDone_names[i], outputListVal_names[i]))

        # make losses as normal for all but outputListVal
        for var_name in output_nodes.keys():
            if not var_name.startswith("outputListVal"):
                data_var_name = "%s_data" % var_name
                data_node = tf.placeholder(tf.int32, shape=[None], name=data_var_name)
                output_node = output_nodes[var_name]
                output_rank = output_node.get_shape().ndims
                if output_rank == 1:
                    output_node = tf.tile(tf.expand_dims(output_node, 0), [tf.shape(data_node)[0], 1])
                loss_node = self.tpt.observe(output_node, data_node,
                                             scope="%s_observe" % var_name)
                output_datas[var_name] = data_node
                loss_nodes[var_name] = loss_node

        # make outputListIsDone-weighted loss for outputListVal
        for outputListIsDone_name, outputListVal_name in name_pairs:
            # we've already made a placeholder for outputListIsDone
            outputListIsDone_data_node = output_datas[outputListIsDone_name]

            # make placeholder for getting list data
            list_data_node = tf.placeholder(tf.int32, shape=[None],
                                            name=outputListVal_name)
            output_datas[outputListVal_name] = list_data_node

            # construct weighted loss and add to loss_nodes
            unweighted_loss_node = self.tpt.observe(output_nodes[outputListVal_name],
                                               list_data_node,
                                               scope="%s_observe" % outputListVal_name)
            weight = 1.0 - tf.to_float(outputListIsDone_data_node)
            loss_node = weight * unweighted_loss_node
            loss_nodes[outputListVal_name] = loss_node

        return (output_datas, loss_nodes)


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
