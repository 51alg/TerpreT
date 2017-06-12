from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

numBlocks = Hyper()
numRegisters = Hyper()
numTimesteps = Hyper()
maxInt = Hyper()

# Inputs and Outputs
boolSize = 2
initial_memory = Input(maxInt)[maxInt]
final_is_halted = Output(boolSize)
final_memory = Output(maxInt)[maxInt]

# Interpreter instruction details / implementations
numInstructions = 16
@Runtime([], maxInt)
def Zero(): return 0
@Runtime([maxInt], maxInt)
def Inc(a): return (a + 1) % maxInt
@Runtime([maxInt, maxInt], maxInt)
def Add(a, b): return (a + b) % maxInt
@Runtime([maxInt, maxInt], maxInt)
def Sub(a, b): return (a - b) % maxInt
@Runtime([maxInt], maxInt)
def Dec(a): return (a - 1) % maxInt
@Runtime([maxInt, maxInt], maxInt)
def LessThan(a, b): return 1 if a < b else 0
@Runtime([maxInt, maxInt], maxInt)
def LessThanOrEqual(a, b): return 1 if a <= b else 0
@Runtime([maxInt, maxInt], boolSize)
def EqualityTest(a, b): return 1 if a == b else 0
@Runtime([maxInt, maxInt], boolSize)
def Min(a, b): return a if a <= b else b
@Runtime([maxInt, maxInt], boolSize)
def Max(a, b): return b if a <= b else a
@Runtime([numRegisters, numRegisters], boolSize)
def EqualityTestReg(a, b): return 1 if a == b else 0
@Runtime([maxInt], boolSize)
def GreaterThanZero(a): return 1 if a > 0 else 0

# Program parameters
instructions = Param(numInstructions)[numBlocks]
thenBlock = Param(numBlocks)[numBlocks]
elseBlock = Param(numBlocks)[numBlocks]
arg1Reg = Param(numRegisters)[numBlocks]
arg2Reg = Param(numRegisters)[numBlocks]
outReg = Param(numRegisters)[numBlocks]
condReg = Param(numRegisters)[numBlocks]

# State of registers, memory and program pointer during execution:
isHalted = Var(boolSize)[numTimesteps + 1]
blockPointer = Var(numBlocks)[numTimesteps + 1]
registers = Var(maxInt)[numTimesteps + 1, numRegisters]
memory = Var(maxInt)[numTimesteps + 1, maxInt]

# Temporary values:
tmpOutput = Var(maxInt)[numTimesteps]
tmpArg1Val = Var(maxInt)[numTimesteps]
tmpArg2Val = Var(maxInt)[numTimesteps]
tmpIsMemoryWrite = Var(boolSize)[numTimesteps]
tmpDoWriteRegister = Var(boolSize)[numTimesteps, numRegisters]
tmpDoWriteMemoryCell = Var(boolSize)[numTimesteps, maxInt]
tmpGoToThenBlock = Var(boolSize)[numTimesteps]

# Copy in inputs and set up initial register values:
isHalted[0].set_to(0)
blockPointer[0].set_to(0)
for a in range(maxInt):
    memory[0, a].set_to(initial_memory[a])
for r in range(numRegisters):
    registers[0, r].set_to(0)

for t in range(numTimesteps):
    if isHalted[t] == 1:
        for a in range(maxInt):
            memory[t + 1, a].set_to(memory[t, a])
        for r in range(numRegisters):
            registers[t + 1, r].set_to(registers[t, r])
        blockPointer[t + 1].set_to(blockPointer[t])
        isHalted[t + 1].set_to(1)
    elif isHalted[t] == 0:
        with blockPointer[t] as blockPointerId:
            # Extract argument values for current block:
            with arg1Reg[blockPointerId] as arg1RegId:
                tmpArg1Val[t].set_to(registers[t, arg1RegId])
            with arg2Reg[blockPointerId] as arg2RegId:
                tmpArg2Val[t].set_to(registers[t, arg2RegId])

            # Execute instruction and put output in temporary variable
            if instructions[blockPointerId] == 0:  # halt
                tmpOutput[t].set_to(tmpArg1Val[t])
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(1)
            elif instructions[blockPointerId] == 1:  # noop
                tmpOutput[t].set_to(tmpArg1Val[t])
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 2:  # write
                tmpOutput[t].set_to(tmpArg1Val[t])  # this will remain unused
                tmpIsMemoryWrite[t].set_to(1)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 3:  # read
                with tmpArg1Val[t] as readAddr:
                    tmpOutput[t].set_to(memory[t, readAddr])
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 4:  # zero
                tmpOutput[t].set_to(0)
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 5:  # inc
                tmpOutput[t].set_to(Inc(tmpArg1Val[t]))
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 6:  # dec
                tmpOutput[t].set_to(Dec(tmpArg1Val[t]))
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 7:  # add
                tmpOutput[t].set_to(Add(tmpArg1Val[t], tmpArg2Val[t]))
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 8:  # sub
                tmpOutput[t].set_to(Sub(tmpArg1Val[t], tmpArg2Val[t]))
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 9:  # lessThan
                tmpOutput[t].set_to(LessThan(tmpArg1Val[t], tmpArg2Val[t]))
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 10:  # lessOrEqualThan
                tmpOutput[t].set_to(LessThanOrEqual(tmpArg1Val[t], tmpArg2Val[t]))
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 11:  # equalTo
                tmpOutput[t].set_to(EqualityTest(tmpArg1Val[t], tmpArg2Val[t]))
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 12:  # one
                tmpOutput[t].set_to(1)
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 13:  # two
                tmpOutput[t].set_to(2)
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 14:  # min
                tmpOutput[t].set_to(Min(tmpArg1Val[t], tmpArg2Val[t]))
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)
            elif instructions[blockPointerId] == 15:  # max
                tmpOutput[t].set_to(Max(tmpArg1Val[t], tmpArg2Val[t]))
                tmpIsMemoryWrite[t].set_to(0)
                isHalted[t + 1].set_to(0)

            # Prepare registers / memory for next timestep:
            if tmpIsMemoryWrite[t] == 0:
                for r in range(numRegisters):
                    tmpDoWriteRegister[t, r].set_to(EqualityTestReg(outReg[blockPointerId], r))
                    if tmpDoWriteRegister[t, r] == 1:
                        registers[t + 1, r].set_to(tmpOutput[t])
                    elif tmpDoWriteRegister[t, r] == 0:
                        registers[t + 1, r].set_to(registers[t, r])
                for a in range(maxInt):
                    memory[t + 1, a].set_to(memory[t, a])
            elif tmpIsMemoryWrite[t] == 1:
                for r in range(numRegisters):
                    registers[t + 1, r].set_to(registers[t, r])
                for a in range(maxInt):
                    tmpDoWriteMemoryCell[t, a].set_to(EqualityTest(tmpArg1Val[t], a))
                    if tmpDoWriteMemoryCell[t, a] == 0:
                        memory[t + 1, a].set_to(memory[t, a])
                    elif tmpDoWriteMemoryCell[t, a] == 1:
                        memory[t + 1, a].set_to(tmpArg2Val[t])

            # Figure out the next block:
            with condReg[blockPointerId] as condRegId:
                tmpGoToThenBlock[t].set_to(GreaterThanZero(registers[t + 1, condRegId]))
                if tmpGoToThenBlock[t] == 1:
                    blockPointer[t + 1].set_to(thenBlock[blockPointerId])
                elif tmpGoToThenBlock[t] == 0:
                    blockPointer[t + 1].set_to(elseBlock[blockPointerId])

final_is_halted.set_to(isHalted[numTimesteps])
for a in range(maxInt):
    final_memory[a].set_to(memory[numTimesteps, a])
