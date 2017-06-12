from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

numTapeSymbols = Hyper()
numHeadStates = Hyper()
tapeLength = Hyper()
numTimesteps = Hyper()
boolSize = 2
numDirections = 3

# Inputs and Output
initial_tape = Input(numTapeSymbols)[tapeLength]
final_is_halted = Output(2)
final_tape = Output(numTapeSymbols)[tapeLength]

# Turing Machine parameters
write = Param(numTapeSymbols)[numHeadStates, numTapeSymbols]
dir = Param(numDirections)[numHeadStates, numTapeSymbols]
newState = Param(numHeadStates)[numHeadStates, numTapeSymbols]

@Runtime([tapeLength, numDirections], tapeLength)
def move(pos, dir):
    if dir == 0:
        return pos
    elif dir == 1:
        return (pos + 1) % tapeLength
    elif dir == 2:
        return (pos - 1) % tapeLength
@Runtime([tapeLength, tapeLength], boolSize)
def EqualityTest(a, b): return 1 if a == b else 0
@Runtime([numHeadStates, numHeadStates], boolSize)
def EqualityTestState(a, b): return 1 if a == b else 0

# State of tape and head during execution:
tape = Var(numTapeSymbols)[numTimesteps, tapeLength]
curPos = Var(tapeLength)[numTimesteps]
curState = Var(numHeadStates)[numTimesteps]
isHalted = Var(boolSize)[numTimesteps]

# Temporary values:
tmpActiveCell = Var(boolSize)[numTimesteps - 1, tapeLength]
tmpCurSymbol = Var(numTapeSymbols)[numTimesteps - 1]

# Constant start state
curPos[0].set_to(0)
curState[0].set_to(1)
isHalted[0].set_to(0)

for p in range(tapeLength):
    tape[0, p].set_to(initial_tape[p])

for t in range(numTimesteps - 1):
    if isHalted[t] == 1:
        for m in range(tapeLength):
            tape[t + 1, m].set_to(tape[t, m])
        curState[t + 1].set_to(curState[t])
        curPos[t + 1].set_to(curPos[t])
        isHalted[t + 1].set_to(isHalted[t])

    elif isHalted[t] == 0:
        with curState[t] as s:
            with curPos[t] as p:
                with tape[t, p] as tt:
                    tmpCurSymbol[t].set_to(write[s, tt])
                    curPos[t + 1].set_to(move(p, dir[s, tt]))
                    curState[t + 1].set_to(newState[s, tt])

        isHalted[t+1].set_to(EqualityTestState(0, curState[t + 1]))

        for m in range(tapeLength):
            tmpActiveCell[t, m].set_to(EqualityTest(m, curPos[t]))
            if tmpActiveCell[t, m] == 1:
                tape[t + 1, m].set_to(tmpCurSymbol[t])
            elif tmpActiveCell[t, m] == 0:
                tape[t + 1, m].set_to(tape[t, m])


final_is_halted.set_to(isHalted[numTimesteps - 1])
for p in range(tapeLength):
    final_tape[p].set_to(tape[numTimesteps - 1, p])
