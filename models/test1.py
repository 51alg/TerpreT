const_T = Hyper()
const_M = Hyper()

@Runtime([const_M, const_M, const_M], const_M)
def Update(prev, cur, offset):
    return (prev + cur + offset) % 2

offset = Param(const_M)
do_anything = Param(2)

initial_tape = Input(const_M)[2]
tape = Var(const_M)[const_T]

for t in range(2):
    tape[t].set_to(initial_tape[t])


for t in range(2, const_T):
    if do_anything == 1:
        tape[t].set_to(Update(tape[t - 2], tape[t - 1], offset))
    elif do_anything == 0:
        tape[t].set_to(tape[t - 1])

final_tape = Output(const_M)
final_tape.set_to(tape[const_T - 1])
