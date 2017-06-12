# Overview
`TerpreT` is a domain-specific language for expressing program synthesis
problems.
It is similar to a probabilistic programming language: a model is composed
of a specification of a program representation (declarations of random
variables) and an interpreter that describes how programs map inputs to
outputs (a model connecting unknowns to observations). The inference task
is to observe a set of input-output examples and infer the underlying
program.

A [tech report](https://arxiv.org/abs/1608.04428) discussing `TerpreT` explains the used techniques
in detail.

# Installation
`TerpreT` itself is entirely implemented in Python and only requires a small
number of Python packages, which can be installed using pip as follows.

```
pip install astunparse h5py docopt 
```

The different `TerpreT` backends and their compilers have additional
requirements:

* `FMGD`: Compilation has no additional requirements, but training needs
  `TensorFlow` to be installed. We have tested on version 0.12, which can
  be installed following the Google-provided
  [instructions](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html).
* `ILP`: TODO
* `SMT`: Compilation requires the `z3` Python bindings.  On Debian and related
  systems (e.g., Ubuntu), installing the package `python-z3` with
  ```
  apt-get install python-z3
  ```
  is sufficient.
  For other systems, follow the instructions in the
  [github repository](https://github.com/Z3Prover/z3).
  The compiled files can be solved by any SMTLIB2-compatible solver, such as
   - [`z3`](https://github.com/Z3Prover/z3)
   - [`CVC`](http://cvc4.cs.nyu.edu/web/)
   - [`Barcelogic`](http://www.cs.upc.edu/~oliveras/bclt-main.html)
  After installation, edit `/TerpreT/lib/config.py` to set `LIB_Z3_PATH` to the path where you just installed it.
* `Sketch`: Compilation has no additional requirements, but solving the sketches
  requires [`Sketch`](https://people.csail.mit.edu/asolar/sketch-1.7.2.tar.gz).

# Usage

We show how to use our different backends on a program induction task. Such
a task is made up of an interpreter model, a choice of model parameters, and
input/output example data. For our example, we use a simple model working
on a finite-length tape, stored in `models/test1.py`.

All executables in `bin/` provide a `--help` option that documents optional
parameters.

## Data

We provide data from the TerpreT tech report in files named
`/TerpreT/data/arxiv_v1/<model>_<problem>_data.json`

To generate more data, data generators are in /TerpreT/data/generators. We recommend storing data in the /TerpreT/data directory.
To populate the directory data with Turing Machine data, run
```
/TerpreT$ cd data
/TerpreT$ generators/make_turing_data.py
```

This will create JSON files with data `turing_*_data.json` and a file of hyperparameters `turing_*_hypers.json`.


## Using the FMGD backend

Compilation combines the interpreter model and chosen model hyperparameters,
producing a computation graph description in TensorFlow. This step is independent of the
input-output data. Data will be incorporated at runtime.
```
TerpreT$ bin/compile_tensorflow.py models/test1.py data/test1_hypers.json
Reading interpreter model from 'models/test1.py'.
Reading model parameters from 'data/test1_hypers.json'.
Outputting to ./compiled/tensorflow_models/test1_hypers_compiled.py
```

We have a TensorFlow model description and are ready to infer the program.
At this point we also incorporate the data:
```
TerpreT$ bin/train.py ./compiled/tensorflow_models/test1_hypers_compiled.py data/test1_data.json --num-restarts 10 --print-params
Construct forward graph... done in 0.48s.
Construct gradient graph... done in 1.45s.
Construct apply gradient graph... done in 0.03s.
Construct forward graph... done in 0.26s.
Construct gradient graph... done in 0.82s.
Construct apply gradient graph... done in 0.03s.
Initializing variables... done in 1.07s.
epoch 100, loss 0.547312, needed 0.19s
epoch 200, loss 0.149809, needed 0.15s
epoch 300, loss 0.0296652, needed 0.15s
epoch 400, loss 0.00564954, needed 0.15s
epoch 500, loss 0.00106832, needed 0.15s
epoch 600, loss 0.000202073, needed 0.15s
epoch 700, loss 3.96957e-05, needed 0.15s
epoch 800, loss 1.10069e-05, needed 0.15s
  Early training termination after 810 epochs because loss 0.000010 is below bound 0.000010.
** PARAMS **
do_anything     : [ 0.000  1.000]
offset  : [ 1.000  0.000]
** DISCRETIZED PARAMS **
do_anything: 1
offset: 0
Training stopped after  5s.
...
```
The `--print-params` option indicates that the resulting solution should be displayed.
Instead of `bin/train.py`, `bin/custom_train.py` can be used, allowing one to modify
many additional training parameters.


## Using the ILP backend
Compilation merges the interpreter model, model hyperparameters and the example data:
TODO: Document this fully
```
TerpreT$ bin/compile_ilp.py models/test1.py data/test1_hypers.json data/test1_data.json
```


## Using the SMT backend
Compilation merges the interpreter model, model hyperparameters and the example data:
```
TerpreT$ bin/compile_smt.py models/test1.py data/test1_hypers.json data/test1_data.json
Reading interpreter model from 'models/test1.py'.
Reading example data from 'data/test1_data.json'.
Reading model parameters for configuration 'T4_M2' from 'data/test1_hypers.json'.
Unrolling execution model.
Generating SMT constraint for I/O example 1.
Generating SMT constraint for I/O example 2.
Generating SMT constraint for I/O example 3.
Writing SMTLIB2 benchmark info to 'compiled/smt2_files/test1-test1_data-train.smt2'.
```

Inference requires a call to an SMT solver (here, `z3`):
```
TerpreT$ z3 compiled/smt2_files/test1-test1_data-train.smt2
sat
(model
  (define-fun tape_2__ex1 () Int
    1)
  (define-fun offset__ex3 () Int
    1)
  (define-fun do_anything__ex2 () Int
    1)
  ...
)
```
The exact output format depends on the used SMT solver, but will generally
look similar to the above. Here, `offset__ex3` is the variable corresponding
to the parameter `offset` (copies for all example
instances will exist, but are guaranteed to have the same value).


## Using the Sketch backend
Compilation merges the interpreter model, model hyperparameters and the example data:
```
TerpreT$ bin/compile_sketch.py models/test1.py data/test1_hypers.json data/test1_data.json
Reading interpreter model from 'models/test1.py'.
Reading example data from 'data/test1_data.json'.
Reading model parameters for configuration 'T4_M2' from 'data/test1_hypers.json'.
Wrote program Sketch to 'compiled/sketches/test1-test1_data-train.sk'.
```

Inference requires a call to the Sketch frontend tool:
```
TerpreT$ sketch compiled/sketches/test1-test1_data-train.sk
SKETCH version 1.6.9
Benchmark = compiled/sketches/test1-test1_data-train.sk
/* BEGIN PACKAGE ANONYMOUS*/
/*test1-t..-train.sk:8*/

void Update1 (int prev, int cur, int offset, ref int _out)/*test1-t..-train.sk:8*/
{
  _out = 0;
  _out = ((prev + cur) + offset) % 2;
  return;
}
/*test1-t..-train.sk:4*/

void _main ()/*test1-t..-train.sk:4*/
{
  int[3][2] initial_tape = {0,0};
  initial_tape[0][0] = 0;
  initial_tape[1][0] = 0;
  initial_tape[0][1] = 1;
  initial_tape[1][1] = 1;
  initial_tape[0][2] = 1;
  initial_tape[1][2] = 0;
  int[3][4] tape = {0,0,0,0};
  int[3] final_tape = {0,0,0};
  for(int input_idx = 0; input_idx < 3; input_idx = input_idx + 1)/*Canonical*/
  {
    for(int t = 0; t < 2; t = t + 1)/*Canonical*/
    {
      tape[t][input_idx] = initial_tape[t][input_idx];
    }
    for(int t_0 = 2; t_0 < 4; t_0 = t_0 + 1)/*Canonical*/
    {
      int _out_s1 = 0;
      Update1(tape[t_0 - 2][input_idx], tape[t_0 - 1][input_idx], 1, _out_s1);
      tape[t_0][input_idx] = _out_s1;
    }
    final_tape[input_idx] = tape[3][input_idx];
  }
  assert ((final_tape[0]) == 0); //Assert at test1-t..-train.sk:39 (-4444533888631374117)
  assert ((final_tape[1]) == 1); //Assert at test1-t..-train.sk:40 (403462684460000275)
  assert ((final_tape[2]) == 1); //Assert at test1-t..-train.sk:41 (-5171897063784164545)
}
/*test1-t..-train.sk:4*/

void main__Wrapper ()  implements main__WrapperNospec/*test1-t..-train.sk:4*/
{
  _main();
}
/*test1-t..-train.sk:4*/

void main__WrapperNospec ()/*test1-t..-train.sk:4*/
{ }
/* END PACKAGE ANONYMOUS*/
[SKETCH] DONE
Total time = 671
```
The output of the Sketch frontend is the completed program sketch, in which
parameters that needed to be inferred are replaced by concrete values.
In the Turing Machine example, these are the `write`, `dir` and `newState`
values shown above.


# FAQ

* Q: How to debug NaNs in FMGD?

  A: A first check is whether you are using the logspace runtime. This is the
  default option in `compile_tensorflow.py`. If you are not, try re-compiling
  with the logspace runtime enabled.

  For other issues, use the command line option `--debug` during training. This
  will cause TensorFlow to raise an error when a nan arises, which will trigger
  pdb and show the name of the variable where the first nan occurred. From the
  error, you can go up in the stack trace to trainer.py where the call to
  `self.sess.run` occurs. The TensorFlow variables are stored in 
  `self.model.params`, `self.model.var_nodes`, and `self.model.param_var_nodes`
  (params after softmaxing). You can issue new calls to `self.sess.run`,
  e.g., `(Pdb) self.sess.run([self.model.params["dir_0_0"]])`.

* Q: Why is performance slower than in the paper?

  A: The default compilation option is to represent all probabilities in
  logspace and do log-sum-exp operations. This comes at a runtime cost (e.g.,
  of about a factor of 4 in the Turing Machine model). To disable the logspace
  runtime use the `--runtime standard` option in the call to `compile_tensorflow.py`.

* Q: Why are there other differences to the paper?

  A: We rewrote the FMGD compiler to support minibatching and curriculum learning,
  and we have not re-run the large hyperparameter searches with the new compiler.
  Thus, the paper reports results with the older version of the FMGD compiler.
  We aim to eventually update the tech report to bring results into sync again.
