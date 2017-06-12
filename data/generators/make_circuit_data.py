#!/usr/bin/env python
'''
Usage:
    make_circuit_data.py [options] [OUTDIR]

Options:
       --seeds INT                 Random seeds to generate data for. [default: 0]
       --num-wires NUMLIST         Number of (input) wires. [default: 4,5,6,8,10,12,15,20]
       --num-gates NUMLIST         Number of gates. [default: 4,5,6,8,10,12,15,20,30]
       --num-examples NUM          Number of examples to generate per model parameter setting & seed. [default: 8]
    -h --help                      Show this help.
'''
from collections import defaultdict
from copy import copy
from docopt import docopt
import json
import numpy as np
import os
import pdb
import re
import sys
import traceback


HYPERS_RE = re.compile("W(\d+)_G(\d+)_O(\d+)")


def make_hypers(wire_num, gate_num, output_num):
    return {'numWires': int(wire_num),
            'numGates': int(gate_num),
            'numOutputs': int(output_num)}


def get_hypers_name(hypers):
    return "W%i_G%i_O%i" % (hypers['numWires'],
                            hypers['numGates'],
                            hypers['numOutputs'])


def get_hypers_from_string(hypers_name):
    match = HYPERS_RE.match(hypers_name)
    (wire_num, gate_num, output_num) = match.groups()
    return make_hypers(wire_num, gate_num, output_num)


def make_shift_ex():
    hypers = make_hypers(4, 5, 3)
    instances = []
    for n in xrange(8):
        in_as_binary = "{0:03b}".format(n)
        in_as_binary = [int(i) for i in in_as_binary]
        out_as_binary = copy(in_as_binary)
        if in_as_binary[0] == 1:
            out_as_binary[1] = in_as_binary[2]
            out_as_binary[2] = in_as_binary[1]
        instances.append({'initial_wires': in_as_binary + [0],
                          'final_wires': out_as_binary})

    return {'batch_name': "train",
            'hypers': get_hypers_name(hypers),
            'instances': instances}


def make_adder_ex(num_wires, num_gates):
    assert num_wires >= 3
    hypers = make_hypers(num_wires, num_gates, 2)
    instances = []
    for n in xrange(8):
        in_as_binary = [int(i) for i in "{0:03b}".format(n)]
        out_as_dec = sum(x == 1 for x in in_as_binary)
        out_as_binary = [int(i) for i in "{0:02b}".format(out_as_dec)]

        instances.append({'initial_wires': in_as_binary + [0] * (num_wires - 3),
                          'final_wires': out_as_binary})

    return {'batch_name': "train",
            'hypers': get_hypers_name(hypers),
            'instances': instances}


def make_full_adder_ex(num_examples, num_wires, num_gates, seed):
    assert num_wires >= 4
    hypers = make_hypers(num_wires, num_gates, 3)
    instances = []
    np.random.seed(seed)
    for _ in xrange(num_examples):
        num = np.random.randint(0, 16)
        in_as_binary = [int(i) for i in "{0:04b}".format(num)]
        a = 2 * in_as_binary[0] + in_as_binary[1]
        b = 2 * in_as_binary[2] + in_as_binary[3]
        res = a + b
        out_as_binary = [int(i) for i in "{0:03b}".format(res)]
        instances.append({'initial_wires': in_as_binary + [0] * (num_wires - 4),
                          'final_wires': out_as_binary})

    return {'batch_name': "train",
            'hypers': get_hypers_name(hypers),
            'instances': instances}


def write_hypers(outdir, hypers):
    if not(os.path.isdir(outdir)):
        os.makedirs(outdir)
    out_file = os.path.join(outdir, "circuit.hypers.json")
    with open(out_file, 'w') as f:
        json.dump({get_hypers_name(hypers): hypers for hypers in hypers}, f, indent=2)


def write_data(outdir, examples, name):
    by_hypers = defaultdict(list)
    for exampleset in examples:
        by_hypers[exampleset['hypers']].append(exampleset)

    if not(os.path.isdir(outdir)):
        os.makedirs(outdir)

    for (hypers_name, examplesetlist) in by_hypers.iteritems():
        out_file = os.path.join(outdir, "circuit_%s_%s.data.json" % (name, hypers_name))
        with open(out_file, 'w') as f:
            json.dump(examplesetlist, f, indent=2)


if __name__ == "__main__":
    args = docopt(__doc__)
    outdir = args.get('OUTDIR', None) or '.'
    seeds = [int(s) for s in args['--seeds'].split(",")]
    wire_nums = [int(n) for n in args['--num-wires'].split(",")]
    gate_nums = [int(n) for n in args['--num-gates'].split(",")]
    example_num = int(args['--num-examples'])

    shift_examples = []
    adder_examples = []
    full_adder_examples = []

    try:
        shift_examples.append(make_shift_ex())
        for wire_num in wire_nums:
            for gate_num in gate_nums:
                adder_examples.append(make_adder_ex(wire_num, gate_num))
                for seed in seeds:
                    full_adder_examples.append(make_full_adder_ex(example_num, wire_num, gate_num, seed))
        hypers_names = set(ex['hypers']
                           for ex in shift_examples + adder_examples + full_adder_examples)
        hypers_list = [get_hypers_from_string(hypers_name)
                       for hypers_name in hypers_names]
        write_hypers(outdir, hypers_list)
        write_data(outdir, shift_examples, "shift")
        write_data(outdir, adder_examples, "adder")
        write_data(outdir, full_adder_examples, "full_adder")
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
