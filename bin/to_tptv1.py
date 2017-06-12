#!/usr/bin/env python
'''
Usage:
    to_tptv1.py [options] MODEL HYPERS DATA [OUTDIR]

Options:
    -h --help                Show this screen.
    --train-batch NAME       Name of the data batch with training data.
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from docopt import docopt
import traceback
import pdb
import ast
from astunparse import unparse
import json
import utils as u
import unroller
import tptv1


if __name__ == "__main__":
    args = docopt(__doc__)
    model_filename = args['MODEL']
    hypers_filename = args.get('HYPERS', None)
    data_filename = args.get('DATA', None)
    train_batch = args.get('--train-batch', 'train') or 'train'

    # Find right data batch:
    with open(data_filename, 'r') as f:
        data_batch_list = json.load(f)
    data_batch = None
    for batch in data_batch_list:
        if batch['batch_name'] == train_batch:
            data_batch = batch
    assert data_batch is not None

    try:
        (parsed_model, data, hypers, out_name) = u.read_inputs(model_filename,
                                                               hypers_filename,
                                                               data_filename,
                                                               train_batch)
        ast = tptv1.translate_to_tptv1(parsed_model, data, hypers)
        print(unparse(ast))
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
