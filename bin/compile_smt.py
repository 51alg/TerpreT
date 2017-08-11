#!/usr/bin/env python
'''
Usage:
    compile_smt.py [options] MODEL HYPERS DATA [OUTDIR]

Options:
    -h --help                Show this screen.
    --train-batch NAME       Name of the data batch with training data.
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lib"))

from docopt import docopt
import traceback
import pdb
import ast
import json
# If you are using Debian/Ubuntu packages, you'll need this:
from z3.z3 import IntVal, Solver, init
import config
init(config.LIB_Z3_PATH)
# If you have installed z3 locally, you will need something like this:
# from z3 import IntVal, Solver, init
# init('/path/to/local/installation/lib/libz3.so')

from terpret_z3 import ToZ3ConstraintsVisitor
import unroller
import utils as u
import tptv1


def compile_smt(model_filename, hypers_filename, data_filename,
                train_batch, out_dir):
    (parsed_model, data, hypers, out_name) = u.read_inputs(model_filename,
                                                           hypers_filename,
                                                           data_filename,
                                                           train_batch)

    print ("Unrolling execution model.")
    parsed_model = u.replace_hypers(parsed_model, hypers)
    unrolled_parsed_model = unroller.unroll_and_flatten(parsed_model,
                                                        do_checks=False,
                                                        print_info=False)
    input_dependents = tptv1.get_input_dependent_vars(unrolled_parsed_model)

    idx = 1
    constraints = []
    z3compilers = []
    for i in data['instances']:
        print ("Generating SMT constraint for I/O example %i." % idx)
        z3compiler = ToZ3ConstraintsVisitor(tag="__ex%i" % idx, variables_to_tag=input_dependents)
        constraints.extend(z3compiler.visit(unrolled_parsed_model))
        for var_name, vals in i.iteritems():
            if vals is None:
                pass
            elif isinstance(vals, list):
                for i, val in enumerate(vals):
                    if val is not None:
                        var_name_item = "%s_%s" % (var_name, i)
                        constraints.append(
                            z3compiler.get_expr(var_name_item) == IntVal(val))
            else:
                constraints.append(
                    z3compiler.get_expr(var_name) == IntVal(vals))
        z3compilers.append(z3compiler)
        idx = idx + 1

    # Unify things:
    z3compiler = z3compilers[0]
    for i in xrange(1, len(z3compilers)):
        z3compilerP = z3compilers[i]
        for param in z3compiler.get_params():
            constraints.append(
                z3compiler.get_expr(param) == z3compilerP.get_expr(param))

    out_file_name = os.path.join(out_dir, out_name + ".smt2")
    print "Writing SMTLIB2 benchmark info to '%s'." % out_file_name
    solver = Solver()
    idx = 0
    for c in constraints:
        # Debugging helper if things unexpectedly end up UNSAT:
        # solver.assert_and_track(c, "c%i" % idx)
        solver.add(c)
        idx = idx + 1
    with open(out_file_name, 'w') as f:
        f.write(solver.to_smt2())
        f.write("(get-model)")
        # Debugging helper if things unexpectedly end up UNSAT:
        # print solver.check()
        # core = solver.unsat_core()
        # print "Size of unsat core: %i" % len(core)
        # idx = 0
        # for c in constraints:
        #    if Bool("c%i" % idx) in core:
        #        print "CORE: %s" % (constraints[idx])
        #    idx = idx + 1

if __name__ == "__main__":
    args = docopt(__doc__)

    source_filename = args['MODEL']
    hypers_filename = args['HYPERS']
    data_filename = args['DATA']
    out_dir = args.get('OUTDIR', None) or "compiled/smt2_files/"
    out_dir = os.path.join(out_dir, "")
    train_batch = args.get('--train-batch', 'train') or 'train'

    try:
        compile_smt(source_filename, hypers_filename, data_filename,
                    train_batch, out_dir)
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
