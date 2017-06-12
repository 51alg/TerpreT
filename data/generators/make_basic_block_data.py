#!/usr/bin/env python
'''
Usage:
    make_basic_block_data.py [options] [OUTDIR]

Options:
       --seeds INT                 Random seeds to generate data for. [default: 0]
       --max-int NUMLIST           Number if integer values. [default: 5,8]
       --num-blocks NUMLIST        Number of basic blocks. [default: 5]
       --num-registers NUMLIST     Number of registers. [default: 2]
       --num-timesteps NUMLIST     Number of timesteps to run. [default: 6,12,20]
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


def make_hypers(max_int, num_blocks, num_registers, num_timesteps):
    return {'maxInt': max_int,
            'numBlocks': num_blocks,
            'numRegisters': num_registers,
            'numTimesteps': num_timesteps}


def get_hypers_name(hypers):
    return "M%i_B%i_R%i_T%i" % (hypers['maxInt'],
                                hypers['numBlocks'],
                                hypers['numRegisters'],
                                hypers['numTimesteps'])


def make_one_access_ex(hypers):
    k = np.random.randint(hypers['maxInt'] - 2)
    initial_memory = [k] + list(np.random.randint(hypers['maxInt'], size=hypers['maxInt'] - 1))
    final_memory = [initial_memory[k + 1]] + list(initial_memory[1:])
    return {'initial_memory': initial_memory,
            'final_memory': final_memory,
            'final_is_halted': 1}


def make_one_decrement_ex(hypers):
    data_length = np.random.randint(1, hypers['maxInt'])
    data = np.random.randint(2, hypers['maxInt'], size=data_length)
    blank_space = [0] * (hypers['maxInt'] - data_length)
    initial_memory = list(data) + blank_space
    final_memory = list(-1 + data) + blank_space
    return {'initial_memory': initial_memory,
            'final_memory': final_memory,
            'final_is_halted': 1}


def make_one_listK_ex(hypers):
    list_length = (hypers['maxInt'] - 1) / 2
    k = np.random.randint(list_length)
    list_data = np.random.randint(hypers['maxInt'], size=list_length)
    # Compute layout in memory for list, making sure that first element is first:
    layout = list(range(1, list_length))
    np.random.shuffle(layout)
    layout = [0] + layout
    initial_memory = [0] * hypers['maxInt']
    initial_memory[0] = k
    for idx in range(list_length):
        initial_memory[1 + 2 * layout[idx]] = list_data[idx]
        if idx < list_length - 1:
            initial_memory[1 + 2 * layout[idx] + 1] = 1 + 2 * layout[idx + 1]
        else:
            initial_memory[1 + 2 * layout[idx] + 1] = 0
    final_memory = list(initial_memory)
    final_memory[0] = list_data[k]
    return {'initial_memory': initial_memory,
            'final_memory': final_memory,
            'final_is_halted': 1}


def make_ex_data(hypers, seed, example_num, make_one_fun):
    np.random.seed(seed)
    instances = [make_one_fun(hypers) for _ in xrange(example_num)]
    return {'batch_name': get_batch_name(seed),
            'hypers': get_hypers_name(hypers),
            'instances': instances}


def write_hypers(outdir, hypers):
    if not(os.path.isdir(outdir)):
        os.makedirs(outdir)
    out_file = os.path.join(outdir, "basic_block.hypers.json")
    with open(out_file, 'w') as f:
        json.dump({get_hypers_name(hypers): hypers for hypers in hypers}, f, indent=2)


def write_data(outdir, examples, name):
    by_hypers = defaultdict(list)
    for exampleset in examples:
        by_hypers[exampleset['hypers']].append(exampleset)

    if not(os.path.isdir(outdir)):
        os.makedirs(outdir)

    for (hypers_name, examplesetlist) in by_hypers.iteritems():
        out_file = os.path.join(outdir, "basic_block_%s_%s.data.json" % (name, hypers_name))
        with open(out_file, 'w') as f:
            json.dump(examplesetlist, f, indent=2)


if __name__ == "__main__":
    args = docopt(__doc__)
    outdir = args.get('OUTDIR', None) or '.'
    seeds = [int(s) for s in args['--seeds'].split(",")]
    max_ints = [int(l) for l in args['--max-int'].split(",")]
    block_nums = [int(n) for n in args['--num-blocks'].split(",")]
    register_nums = [int(n) for n in args['--num-registers'].split(",")]
    timestep_nums = [int(n) for n in args['--num-timesteps'].split(",")]
    example_num = int(args['--num-examples'])

    hypers_list = []
    access_examples = []
    decrement_examples = []
    listK_examples = []

    try:
        for max_int in max_ints:
            for num_blocks in block_nums:
                for num_registers in register_nums:
                    for num_timesteps in timestep_nums:
                        hypers = make_hypers(max_int, num_blocks, num_registers, num_timesteps)
                        hypers_list.append(hypers)
                        for seed in seeds:
                            access_examples.append(make_ex_data(hypers, seed, example_num, make_one_access_ex))
                            decrement_examples.append(make_ex_data(hypers, seed, example_num, make_one_decrement_ex))
                            listK_examples.append(make_ex_data(hypers, seed, example_num, make_one_listK_ex))
        write_hypers(outdir, hypers_list)
        write_data(outdir, access_examples, "access")
        write_data(outdir, decrement_examples, "decrement")
        write_data(outdir, listK_examples, "listK")
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
