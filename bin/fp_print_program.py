#!/usr/bin/env python2
# -*- coding: utf-8 -*-

'''
Usage:
    fp_print_program.py [options] DATAFILE

Options:
    -h --help                Show this screen.
    -e --epsilon FLOAT       Probability bound under which we filter values. [default: 1e-2]
    -d --discretize          Discretize all parameters, i.e., only display highest value.
       --print-trace NUM     Print trace for first NUM examples.
    --debug                  Enable debug routines. [default: False]
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from fp_printer_utils import *
from docopt import docopt
import traceback
import pdb
import h5py


def main(args):
    datafile = args['DATAFILE']

    par_prob_bound = float(args.get('--epsilon', 1e-2))
    if args.get('--discretize', False):
        par_prob_bound = None

    raw_data = h5py.File(datafile, 'r')
    hypers = {k: raw_data['model_hypers'].attrs[k] for k in raw_data['model_hypers'].attrs.keys()}

    if 'numTimesteps' in hypers:
        if 'maxScalar' in hypers:
            printer = AssemblyPrinter(par_prob_bound, raw_data, hypers)
        else:
            printer = AssemblyFixedAllocPrinter(par_prob_bound, raw_data, hypers)
    elif 'extraRegisterNum' in hypers:
        if 'loopBodyLength' in hypers:
            if 'maxInt' in hypers:
                printer = AssemblyLoopTypedPrinter(par_prob_bound, raw_data, hypers)
            else:
                printer = AssemblyLoopUntypedPrinter(par_prob_bound, raw_data, hypers)
        else:
            if 'maxInt' in hypers:
                printer = CombinatorTypedMutablePrinter(par_prob_bound, raw_data, hypers)
            else:
                printer = CombinatorUntypedMutablePrinter(par_prob_bound, raw_data, hypers)
    else:
        if 'maxInt' in hypers:
            printer = CombinatorTypedImmutablePrinter(par_prob_bound, raw_data, hypers)
        else:
            printer = CombinatorUntypedImmutablePrinter(par_prob_bound, raw_data, hypers)

    printer.print_program()

    trace_to_print = args.get('--print-trace', None)
    if trace_to_print is not None:
        printer.print_trace(int(trace_to_print))

if __name__ == '__main__':
    args = docopt(__doc__)
    try:
        main(args)
    except:
        if args.get('--debug', False):
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise

