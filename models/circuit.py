from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

numGates = Hyper()
numWires = Hyper()
numOutputs = Hyper()
numGateTypes = 5

@Runtime([2, 2], 2)
def AND(a, b): return int(a and b)

@Runtime([2, 2], 2)
def OR(a, b): return int(a or b)

@Runtime([2, 2], 2)
def XOR(a, b): return int(a ^ b)

@Runtime([2], 2)
def NOT(a): return int(not a)

@Runtime([2], 2)
def NOP(a): return a

@Runtime([numWires, numWires], 2)
def equalityTest(a, b): return 1 if a == b else 0

gate = Param(numGateTypes)[numGates]
in1 = Param(numWires)[numGates]
in2 = Param(numWires)[numGates]
out = Param(numWires)[numGates]

initial_wires = Input(2)[numWires]
final_wires = Output(2)[numOutputs]

wires = Var(2)[numGates + 1, numWires]
tmpOutput = Var(2)[numGates]
tmpDoWrite = Var(2)[numGates, numWires]
tmpArg1 = Var(2)[numGates]
tmpArg2 = Var(2)[numGates]

for w in range(numWires):
    wires[0, w].set_to(initial_wires[w])

for g in range(numGates):
    with in1[g] as i1:
        tmpArg1[g].set_to(wires[g, i1])

    with in2[g] as i2:
        tmpArg2[g].set_to(wires[g, i2])

    if gate[g] == 0:
        tmpOutput[g].set_to(AND(tmpArg1[g], tmpArg2[g]))
    elif gate[g] == 1:
        tmpOutput[g].set_to(OR(tmpArg1[g], tmpArg2[g]))
    elif gate[g] == 2:
        tmpOutput[g].set_to(XOR(tmpArg1[g], tmpArg2[g]))
    elif gate[g] == 3:
        tmpOutput[g].set_to(NOT(tmpArg1[g]))
    elif gate[g] == 4:
        tmpOutput[g].set_to(NOP(tmpArg1[g]))

    for w in range(numWires):
        tmpDoWrite[g, w].set_to(equalityTest(out[g], w))
        if tmpDoWrite[g, w] == 1:
            wires[g + 1, w].set_to(tmpOutput[g])
        elif tmpDoWrite[g, w] == 0:
            wires[g + 1, w].set_to(wires[g, w])

for w in range(numOutputs):
    final_wires[w].set_to(wires[numGates, w])
