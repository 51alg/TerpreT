#!/usr/bin/env python
'''
Usage:
    make_turing_data.py [options] [OUTDIR]

Options:
       --seeds INT                 Random seeds to generate data for. [default: 0]
       --tape-length NUMLIST       Length of tape. [default: 5]
       --num-tape-symbols NUMLIST  Number of tape symbols. [default: 3]
       --num-head-states NUMLIST   Number of head states. [default: 2,3,4,6,8,10]
       --num-timesteps NUMLIST     Number of timesteps to run. [default: 6,7,8,10,15,20]
       --num-examples NUM          Number of examples to generate per model parameter setting & seed. [default: 5]
    -h --help                      Show this help.
'''
from collections import defaultdict
from docopt import docopt
import json
import numpy as np
import os
import pdb
import sys
import traceback


def get_batch_name(seed):
    if seed == 0:
        return "train"
    else:
        return "seed__%i" % seed


def get_hypers_name(hypers):
    return "L%i_T%i_H%i_S%i" % (hypers['tapeLength'],
                                hypers['numTimesteps'],
                                hypers['numHeadStates'],
                                hypers['numTapeSymbols'])


def make_one_prepend_ex(hypers):
    assert hypers['numTapeSymbols'] >= 2
    data_len = np.random.randint(1, hypers['tapeLength'] - 1)
    initial_blanks = [hypers['numTapeSymbols'] - 1] * (hypers['tapeLength'] - data_len - 1)
    final_blanks = [hypers['numTapeSymbols'] - 1] * (hypers['tapeLength'] - data_len - 2)
    initial_data = [np.random.randint(0, hypers['numTapeSymbols'] - 1)
                    for _ in range(data_len + 1)]
    final_data = [0] + initial_data
    return {'initial_tape': initial_data + initial_blanks,
            'final_tape': final_data + final_blanks,
            'final_is_halted': 1}


def make_one_invert_ex(hypers):
    assert hypers['numTapeSymbols'] == 3
    data_len = np.random.randint(1, hypers['tapeLength'] - 1)
    blanks = [2] * (hypers['tapeLength'] - data_len - 1)
    initial_data = [np.random.randint(0, 2)
                    for _ in range(data_len + 1)]
    final_data = [1 - c for c in initial_data]
    return {'initial_tape': initial_data + blanks,
            'final_tape': final_data + blanks,
            'final_is_halted': 1}


def make_one_dec_ex(hypers):
    assert hypers['numTapeSymbols'] == 3
    num_bits = np.random.randint(1, hypers['tapeLength'])
    blanks = [2] * (hypers['tapeLength'] - num_bits)
    decimal_input = np.random.randint(2**(num_bits - 1), 2**num_bits)
    binary_input = ("{0:0" + str(num_bits) + "b}").format(decimal_input)
    binary_final = ("{0:0" + str(num_bits) + "b}").format(decimal_input - 1)
    return {'initial_tape': [int(b) for b in binary_input] + blanks,
            'final_tape': [int(b) for b in binary_final] + blanks,
            'final_is_halted': 1}


def make_ex_data(hypers, seed, example_num, make_one_fun):
    np.random.seed(seed)
    instances = [make_one_fun(hypers) for _ in xrange(example_num)]
    return {'batch_name': get_batch_name(seed),
            'hypers': get_hypers_name(hypers),
            'instances': instances}


def make_hypers(tape_len, symbol_num, state_num, timestep_num):
    return {'tapeLength': tape_len,
            'numTimesteps': timestep_num,
            'numHeadStates': state_num,
            'numTapeSymbols': symbol_num}


def write_hypers(outdir, hypers, name):
    if not(os.path.isdir(outdir)):
        os.makedirs(outdir)
    out_file = os.path.join(outdir, "turing_%s_hypers.json" % name)
    print out_file
    with open(out_file, 'w') as f:
        json.dump({get_hypers_name(hypers): hypers}, f, indent=2)
        # {get_hypers_name(hypers): hypers for hypers in hypers}, 


def write_data(outdir, examples, name):
    by_hypers = defaultdict(list)
    for exampleset in examples:
        by_hypers[exampleset['hypers']].append(exampleset)

    if not(os.path.isdir(outdir)):
        os.makedirs(outdir)

    for (hypers_name, examplesetlist) in by_hypers.iteritems():
        out_file = os.path.join(outdir, "turing_%s_data.json" % (name))
        print out_file
        with open(out_file, 'w') as f:
            json.dump(examplesetlist, f, indent=2)


if __name__ == "__main__":
    args = docopt(__doc__)
    outdir = args.get('OUTDIR', None) or '.'
    seeds = [int(s) for s in args['--seeds'].split(",")]
    tape_lengths = [int(l) for l in args['--tape-length'].split(",")]
    symbol_nums = [int(n) for n in args['--num-tape-symbols'].split(",")]
    state_nums = [int(n) for n in args['--num-head-states'].split(",")]
    timestep_nums = [int(n) for n in args['--num-timesteps'].split(",")]
    example_num = int(args['--num-examples'])

    hypers_list = []
    invert_examples = []
    prepend_examples = []
    decrement_examples = []

    try:
        hypers_dict = {
            "invert": (5, 3, 2, 6),  # invert
            "prepend": (5, 3, 3, 6),  # prepend zero
            "decrement": (5, 3, 3, 9),  # binary decrement
        }
        
        for name, hyper_values in hypers_dict.iteritems():
            hypers = make_hypers(*hyper_values)

            if name == "invert":
                make_one_fn = make_one_invert_ex
            elif name == "prepend":
                make_one_fn = make_one_prepend_ex
            elif name == "decrement":
                make_one_fn = make_one_dec_ex
            else:
                assert False

            for seed in seeds:
                example = make_ex_data(hypers, seed, example_num, make_one_fn)
                write_hypers(outdir, hypers, name)
                write_data(outdir, [example], name)

    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
