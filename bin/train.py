#!/usr/bin/env python
'''
Usage:
    train.py [options] COMPILED_MODEL DATA [TRAIN_HYPERS]

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
    --num-restarts K         Number of independent runs [default: 1]
    --seed S                 Random number seed [default: 0]
    --debug                  Enable debug routines. [default: False]
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from docopt import docopt
import json
import imp
import pdb
import traceback
import h5py
import numpy as np
import time
from data import Data
from trainer import Trainer

def store_results_to_hdf5(filename, trainer, batch_name, store_vars=False, restart_idx=0):
    filename = filename + "_restart%s" % restart_idx
    h5f = h5py.File(filename, 'w', libver='latest')

    hypers_group = h5f.create_group('model_hypers')
    for (hyper_name, hyper_value) in trainer.model_hypers.values()[0].iteritems():
        hypers_group.attrs[hyper_name] = hyper_value

    params_group = h5f.create_group('parameters')
    for (param_name, param_value) in trainer.param_values().iteritems():
        prob_param_values = np.exp(param_value) / np.sum(np.exp(param_value))
        params_group.create_dataset(param_name, data=prob_param_values)

    vars_group = h5f.create_group('variables')
    for (var_name, var_value) in trainer.var_values(batch_name).iteritems():
        normalised_var_values = trainer.tpt.normalize(var_value)
        vars_group.create_dataset(var_name, data=normalised_var_values)

    loss_group = h5f.create_group('losses')
    loss_group.create_dataset('train_loss_values', data=trainer.train_loss_values)
    loss_contribution_group = loss_group.create_group('final_loss_contribution')
    for loss_node_name, loss in trainer.get_loss_breakdown(batch_name).iteritems():
        loss_contribution_group.create_dataset(loss_node_name, data=loss)

    h5f.close()    

def load_model(model_filename):
    module_name = os.path.basename(model_filename).replace(".py", "")
    module = imp.load_source(module_name, model_filename)
    return module.Model(), module

def load_hypers(hypers_filename):
    with open(hypers_filename, 'r') as f:
        hypers_json = json.load(f)
    return hypers_json

def load_trainer(args, trainer, data, seed):
    model_filename = args['COMPILED_MODEL']
    data_filename = args['DATA']

    model, model_module = load_model(model_filename)
    model_hypers = model.get_hypers()
    if 'TRAIN_HYPERS' in args and args['TRAIN_HYPERS'] is not None:
        train_hypers = load_hypers(args['TRAIN_HYPERS'])
    else:
        train_hypers = trainer.default_train_hypers()

    hypers_overrides = args.get('--override-hypers', None)
    if hypers_overrides is not None:
        hypers_overrides = json.loads(hypers_overrides)
        for (hyper, value) in hypers_overrides.iteritems():
            train_hypers[hyper] = value

    loaded_data = data(data_filename)

    return trainer(model, model_module, model_hypers, train_hypers, loaded_data, seed=seed, do_debug=args['--debug'])

def load_and_run(args, trainerClass):
    start_time = time.time()
    seed = int(args.get('--seed', 0))
    trainer = load_trainer(args, trainerClass, Data, seed)
    train_batch_name = args.get('--train-batch', None) or "train"
    validation_batch_name = args.get('--validation-batch', None)
    test_batch_name = args.get('--test-batch', None)
    print_params = args.get('--print-params', False) or False
    print_loss_breakdown = args.get('--print-loss-breakdown', False) or False
    num_restarts = int(args.get('--num-restarts', 1))
    
    
    for i in xrange(num_restarts):
        (params, discretized_params) = trainer.train(train_batch_name,
                                                     validation_batch_name=validation_batch_name,
                                                     test_batch_name=test_batch_name,
                                                     print_params=print_params,
                                                     print_final_loss_breakdown=print_loss_breakdown)
    
        if '--store-data' in args and args['--store-data'] is not None:
            store_results_to_hdf5(args['--store-data'], trainer, train_batch_name, restart_idx=i)
        print ("Training stopped after %2.fs." % (time.time() - start_time)) 

if __name__ == "__main__":
    args = docopt(__doc__)
    try:
        load_and_run(args, Trainer)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
