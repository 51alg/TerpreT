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
numInstructions = 9
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
@Runtime([maxInt, maxInt], boolSize)
def EqualityTest(a, b): return 1 if a == b else 0
@Runtime([numRegisters, numRegisters], boolSize)
def EqualityTestReg(a, b): return 1 if a == b else 0
@Runtime([maxInt], boolSize)
def GreaterThanZero(a): return 1 if a > 0 else 0

# Program parameters
numBlocksWithHalt = numBlocks + 1
instructions = Param(numInstructions)[numBlocksWithHalt]
thenBlock = Param(numBlocksWithHalt)[numBlocksWithHalt]
elseBlock = Param(numBlocksWithHalt)[numBlocksWithHalt]
arg1Reg = Param(numRegisters)[numBlocksWithHalt]
arg2Reg = Param(numRegisters)[numBlocksWithHalt]
outReg = Param(numRegisters)[numBlocksWithHalt]
condReg = Param(numRegisters)[numBlocksWithHalt]

# State of registers, memory and program pointer during execution:
blockPointer = Var(numBlocksWithHalt)[numTimesteps + 1]
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

# Set up halting block:
instructions[0].set_to_constant(0)  # MAKE SURE THIS IS POINTING TO NOOP
thenBlock[0].set_to_constant(0)
elseBlock[0].set_to_constant(0)


# Copy in inputs and set up initial register values:
blockPointer[0].set_to(1)
for a in range(maxInt):
    memory[0, a].set_to(initial_memory[a])
for r in range(numRegisters):
    registers[0, r].set_to(0)

for t in range(numTimesteps):
    with blockPointer[t] as blockPointerId:
        # Extract argument values for current block:
        with arg1Reg[blockPointerId] as arg1RegId:
            tmpArg1Val[t].set_to(registers[t, arg1RegId])
        with arg2Reg[blockPointerId] as arg2RegId:
            tmpArg2Val[t].set_to(registers[t, arg2RegId])

        # Execute instruction and put output in temporary variable
        if instructions[blockPointerId] == 0:  # noop
            tmpOutput[t].set_to(tmpArg1Val[t])
            tmpIsMemoryWrite[t].set_to(0)
        elif instructions[blockPointerId] == 1:  # write
            tmpOutput[t].set_to(tmpArg1Val[t])  # this will remain unused
            tmpIsMemoryWrite[t].set_to(1)
        elif instructions[blockPointerId] == 2:  # read
            with tmpArg1Val[t] as readAddr:
                tmpOutput[t].set_to(memory[t, readAddr])
            tmpIsMemoryWrite[t].set_to(0)
        elif instructions[blockPointerId] == 3:  # zero
            tmpOutput[t].set_to(0)
            tmpIsMemoryWrite[t].set_to(0)
        elif instructions[blockPointerId] == 4:  # inc
            tmpOutput[t].set_to(Inc(tmpArg1Val[t]))
            tmpIsMemoryWrite[t].set_to(0)
        elif instructions[blockPointerId] == 5:  # dec
            tmpOutput[t].set_to(Dec(tmpArg1Val[t]))
            tmpIsMemoryWrite[t].set_to(0)
        elif instructions[blockPointerId] == 6:  # add
            tmpOutput[t].set_to(Add(tmpArg1Val[t], tmpArg2Val[t]))
            tmpIsMemoryWrite[t].set_to(0)
        elif instructions[blockPointerId] == 7:  # sub
            tmpOutput[t].set_to(Sub(tmpArg1Val[t], tmpArg2Val[t]))
            tmpIsMemoryWrite[t].set_to(0)
        elif instructions[blockPointerId] == 8:  # lessThan
            tmpOutput[t].set_to(LessThan(tmpArg1Val[t], tmpArg2Val[t]))
            tmpIsMemoryWrite[t].set_to(0)

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

if blockPointer[numTimesteps] == 0:
    final_is_halted.set_to(1)
else:
    final_is_halted.set_to(0)

for a in range(maxInt):
    final_memory[a].set_to(memory[numTimesteps, a])
