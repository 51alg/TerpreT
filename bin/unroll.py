#!/usr/bin/env python
'''
Usage:
    unroll.py [options] MODEL [HYPERS]

Options:
    -h --help                Show this screen.
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from docopt import docopt
import ast
import astunparse
import copy
import error
from error import check_type
import pdb
import itertools
import traceback
import json

import utils as u
import unroller


def run(source_filename, hypers_filename=None):
    source = open(source_filename, 'r').read()

    if hypers_filename is None:
        hypers_list = {"default": ""}
    else:
        with open(hypers_filename, 'r') as f:
            hypers_list = json.load(f)

    for hypers_name, hypers in hypers_list.iteritems():
        module = ast.parse(source)
        if hypers != "":
            module = u.replace_hypers(module, hypers)
        unrolled_node = unroller.unroll_and_flatten(module, do_checks=True)

        print(astunparse.unparse(unrolled_node))
        # Only do the first one...
        break

    return None

if __name__ == "__main__":
    args = docopt(__doc__)
    model_filename = args['MODEL']
    hypers_filename = args.get('HYPERS', None)

    try:
        run(model_filename, hypers_filename=hypers_filename)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
