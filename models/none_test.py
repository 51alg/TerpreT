# A model to demonstrate the output=null feature
#
# The model handles n_reg=2 registers and repeatedly adds both together storing the result
# in a register specified by the params to learn. The data is engineered such that the correct
# program must output 3*init_reg[0] into reg[0] and we don't care about the value of reg[1]
# (note that reg[1] has to be used to accumulate values to solve the task)
#
# compiling:
#   $ python compile_tensorflow.py ../models/none_test.py ../data/none_test_hypers.json ../compiled/
#
# running: (tf version 0.12)
#   $ python train.py ../compiled/none_test_hypers_compiled.py ../data/none_test.json
#      Construct forward graph... done in 0.84s.
#      Construct gradient graph... done in 2.64s.
#      Construct apply gradient graph... done in 0.02s.
#      Initializing variables... done in 0.41s.
#      {'learning_rate': 0.01,
#      'learning_rate_decay': 0.9,
#      'minibatch_size': 10,
#      'momentum': 0.0,
#      'num_epochs': 2000,
#      'optimizer': 'rmsprop',
#      'print_frequency': 100,
#      'stop_below_loss': 1e-05,
#      'validate_frequency': 100}
#     epoch 100, loss 0.741504, needed 0.90s
#     epoch 200, loss 0.15521, needed 0.52s
#     epoch 300, loss 0.0298771, needed 0.53s
#     epoch 400, loss 0.00566099, needed 0.53s
#     epoch 500, loss 0.00106969, needed 0.55s
#     epoch 600, loss 0.00020335, needed 0.60s
#     epoch 700, loss 4.357e-05, needed 0.59s
#     epoch 800, loss 1.54375e-05, needed 0.53s
#       Early training termination after 871 epochs because loss 0.000010 is below bound 0.000010.
#     Training stopped after 10s.




n_reg = 2
max_int = 10
n_time_steps = Hyper()

# I/O
init_reg = Input(max_int)[n_reg]

@Runtime([max_int, max_int], max_int)
def Add(x, y):
    return (x + y) % 10

out_reg = Param(n_reg)[n_time_steps]
reg = Var(max_int)[n_reg, n_time_steps]

for r in range(n_reg):
    reg[r, 0].set_to(init_reg[r])

for t in range(n_time_steps - 1):
    for r in range(n_reg):
        if out_reg[t] == r:
            reg[r, t+1].set_to(Add(reg[0, t], reg[1, t]))
        else:
            reg[r, t+1].set_to(reg[r, t])


final_reg = Output(max_int)[n_reg]
for r in range(n_reg):
    final_reg[r].set_to(reg[r, n_time_steps - 1])
