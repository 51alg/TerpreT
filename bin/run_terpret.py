#!/usr/bin/env python

'''
Usage:
    run_stored_terpret.py [options] MODEL DATA RESULT

Options:
    -h --help             Show this screen.
    --test-batches NAMES  Names of the data batches with test data.
'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from docopt import docopt
import traceback
import pdb
import terpret_run_utils as runu

if __name__ == '__main__':
    args = docopt(__doc__)

    try:
        model_filename = args['MODEL']
        data_filename = args['DATA']
        result_filename = args['RESULT']
        test_batches = args.get('--test-batches', None)
        runu.test(result_filename, model_filename, data_filename, test_batches)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
