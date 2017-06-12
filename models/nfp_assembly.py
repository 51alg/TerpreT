from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

maxScalar = Hyper()
numRegisters = Hyper()
programLen = Hyper()
numTimesteps = Hyper()
inputNum = Hyper()
inputStackSize = Hyper()

numRegInstrs = 13
numBranchInstrs = 2
numInstrTypes = 3
numInstructions = numBranchInstrs + numRegInstrs + 1

mutableStackSize = maxScalar - inputStackSize - 1

# Inputs
inputRegVal = Input(maxScalar)[inputNum]
inputStackCarVal = Input(maxScalar)[inputStackSize]
inputStackCdrVal = Input(maxScalar)[inputStackSize]

# Outputs
expectListOutput = Input(2)
outputTermState = Output(2)
outputRegVal = Output(maxScalar)
outputListVal = Output(maxScalar)[maxScalar]
outputListIsDone = Output(2)[maxScalar]

# Runtime state
stackCarValue = Var(maxScalar)[numTimesteps + 1, mutableStackSize]
stackCdrValue = Var(maxScalar)[numTimesteps + 1, mutableStackSize]
stackPtr      = Var(mutableStackSize)[numTimesteps + 1]

registers = Var(maxScalar)[numTimesteps + 1, numRegisters]

instrPtr = Var(programLen)[numTimesteps + 1]
returnValue = Var(maxScalar)[numTimesteps + 1]
isHalted = Var(2)[numTimesteps + 1]

# Program
# Register Instructions: cons, car, cdr, zero/nil, add, inc, eq, gt, and, dec, or
# Branch Instructions: jz, jnz, halt
instructions = Param(numInstructions)[programLen]
arg1s        = Param(numRegisters)[programLen]
arg2s        = Param(numRegisters)[programLen]
outs         = Param(numRegisters)[programLen]
branchAddr   = Param(programLen)[programLen]

# Temporary values
tmpArg1Val       = Var(maxScalar)[numTimesteps]
tmpArg2Val       = Var(maxScalar)[numTimesteps]
tmpOutVal        = Var(maxScalar)[numTimesteps]
tmpArg1DerefCarValue = Var(maxScalar)[numTimesteps]
tmpArg1DerefCdrValue = Var(maxScalar)[numTimesteps]
tmpBranchCond    = Var(2)[numTimesteps]

tmpDoPushStack   = Var(2)[numTimesteps]
tmpDoWriteStack  = Var(2)[numTimesteps, maxScalar]

tmpIsRegInstr    = Var(2)[numTimesteps]
tmpRegInstr      = Var(numRegInstrs)[numTimesteps]
tmpDoWriteReg    = Var(2)[numTimesteps, numRegisters]

@Runtime([maxScalar, maxScalar], maxScalar)
def Add(x, y): return (x + y) % maxScalar
@Runtime([maxScalar], maxScalar)
def Inc(x): return (x + 1) % maxScalar
@Runtime([maxScalar], maxScalar)
def Dec(x): return (x - 1) % maxScalar  # Python normalizes to 0..maxScalar
@Runtime([maxScalar, maxScalar], maxScalar)
def EqTest(a, b): return 1 if a == b else 0
@Runtime([maxScalar, maxScalar], maxScalar)
def GtTest(a, b): return 1 if a > b else 0
@Runtime([maxScalar, maxScalar], maxScalar)
def And(a, b): return 1 if a == 1 and b == 1 else 0
@Runtime([maxScalar, maxScalar], maxScalar)
def Or(a, b): return 1 if a == 1 or b == 1 else 0
@Runtime([maxScalar], 2)
def ScalarIsZero(x): return 1 if x == 0 else 0

# Modeling helper functions, not actual instructions:
@Runtime([programLen, programLen], 2)
def InstrPtrEquality(a, b): return 1 if a == b else 0
@Runtime([numRegisters, numRegisters], 2)
def RegisterEquality(a, b): return 1 if a == b else 0
@Runtime([programLen], programLen)
def IncInstrPtr(x): return programLen - 1 if x + 1 >= programLen else x + 1
@Runtime([numInstructions], 2)
def RequiresStackPush(instrIndex): return 1 if instrIndex == 0 else 0
@Runtime([numInstructions], 2)
def IsRegInstr(x): return 1 if x < numRegInstrs else 0
@Runtime([numInstructions], numRegInstrs)
def ToRegInstr(x): return x if x < numRegInstrs else 0
@Runtime([numInstructions], 2)
def IsCons(x): return 1 if x == 0 else 0
@Runtime([mutableStackSize], maxScalar)
def StackPtrToScalar(x): return x + inputStackSize + 1
@Runtime([mutableStackSize], mutableStackSize)
def IncStackPtr(x): return (x + 1) % (mutableStackSize)
@Runtime([mutableStackSize, mutableStackSize], 2)
def PtrEquality(a, b): return 1 if a == b else 0

# Copy input registers to main registers.
for i in range(inputNum):
    registers[0, i].set_to(inputRegVal[i])
for i in range(inputNum, numRegisters):
    registers[0, i].set_to(0)

# Copy input stack to main stack.
for i in range(0, mutableStackSize):
    stackCarValue[0, i].set_to(0)
    stackCdrValue[0, i].set_to(0)
stackPtr[0].set_to(0)

# Start with first instruction, program starts not-halted.
instrPtr[0].set_to(0)
isHalted[0].set_to(0)
returnValue[0].set_to(0)

for t in range(numTimesteps):  # !! factor: numTimesteps
    if isHalted[t] == 0:
        with instrPtr[t] as ip:  # !! factor: numTimesteps * numInstructions
            instruction = instructions[ip]
            arg1Val = tmpArg1Val[t]
            arg2Val = tmpArg2Val[t]
            with arg1s[ip] as r:  # !! factor: numTimesteps * numInstructions * numRegisters
                arg1Val.set_to(registers[t, r])
            with arg2s[ip] as r:  # !! factor: numTimesteps * numInstructions * numRegisters
                arg2Val.set_to(registers[t, r])

            with arg1Val as p:  # !! factor: numTimesteps * numInstructions * maxScalar
                if p == 0:
                    tmpArg1DerefCarValue[t].set_to(0)
                    tmpArg1DerefCdrValue[t].set_to(0)
                elif p <= inputStackSize:
                    tmpArg1DerefCarValue[t].set_to(inputStackCarVal[p - 1])
                    tmpArg1DerefCdrValue[t].set_to(inputStackCdrVal[p - 1])
                else:
                    tmpArg1DerefCarValue[t].set_to(stackCarValue[t, p - inputStackSize - 1])
                    tmpArg1DerefCdrValue[t].set_to(stackCdrValue[t, p - inputStackSize - 1])

            # Build registers for next timestep, where branching instructions don't do anything:
            tmpIsRegInstr[t].set_to(IsRegInstr(instruction))
            if tmpIsRegInstr[t] == 0:
                for r in range(numRegisters):  # !! factor: numTimesteps * numInstructions * numRegisters
                    registers[t+1, r].set_to(registers[t, r])
            elif tmpIsRegInstr[t] == 1:
                tmpRegInstr[t].set_to(ToRegInstr(instruction))
                if tmpRegInstr[t] == 0:  # cons
                    tmpOutVal[t].set_to(StackPtrToScalar(stackPtr[t]))
                elif tmpRegInstr[t] == 1:  # car
                    tmpOutVal[t].set_to(tmpArg1DerefCarValue[t])
                elif tmpRegInstr[t] == 2:  # cdr
                    tmpOutVal[t].set_to(tmpArg1DerefCdrValue[t])
                elif tmpRegInstr[t] == 3:  # zero/nil
                    tmpOutVal[t].set_to(0)
                elif tmpRegInstr[t] == 4:  # add
                    tmpOutVal[t].set_to(Add(arg1Val, arg2Val))
                elif tmpRegInstr[t] == 5:  # inc
                    tmpOutVal[t].set_to(Inc(arg1Val))
                elif tmpRegInstr[t] == 6:  # eq
                    tmpOutVal[t].set_to(EqTest(arg1Val, arg2Val))
                elif tmpRegInstr[t] == 7:  # gt
                    tmpOutVal[t].set_to(GtTest(arg1Val, arg2Val))
                elif tmpRegInstr[t] == 8:  # and
                    tmpOutVal[t].set_to(And(arg1Val, arg2Val))
                elif tmpRegInstr[t] == 9:  # one/true
                    tmpOutVal[t].set_to(1)
                elif tmpRegInstr[t] == 10:  # noop/copy
                    tmpOutVal[t].set_to(arg1Val)
                elif tmpRegInstr[t] == 11:  # dec
                    tmpOutVal[t].set_to(Dec(arg1Val))
                elif tmpRegInstr[t] == 12:  # or
                    tmpOutVal[t].set_to(Or(arg1Val, arg2Val))

                for r in range(numRegisters):  # !! factor: numTimesteps * numInstructions * numRegisters
                    tmpDoWriteReg[t, r].set_to(RegisterEquality(outs[ip], r))
                    if tmpDoWriteReg[t, r] == 1:
                        registers[t+1, r].set_to(tmpOutVal[t])
                    elif tmpDoWriteReg[t, r] == 0:
                        registers[t+1, r].set_to(registers[t, r])

            # Build stack for next timestep, only Cons changes anything
            tmpDoPushStack[t].set_to(IsCons(instruction))
            if tmpDoPushStack[t] == 1:
                for p in range(mutableStackSize):  # !! factor: numTimesteps * numInstructions * maxScalar
                    tmpDoWriteStack[t, p].set_to(PtrEquality(stackPtr[t], p))
                    if tmpDoWriteStack[t, p] == 1:
                        stackCarValue[t+1, p].set_to(arg1Val)
                        stackCdrValue[t+1, p].set_to(arg2Val)
                    elif tmpDoWriteStack[t, p] == 0:
                        stackCarValue[t+1, p].set_to(stackCarValue[t, p])
                        stackCdrValue[t+1, p].set_to(stackCdrValue[t, p])
                stackPtr[t+1].set_to(IncStackPtr(stackPtr[t]))
            elif tmpDoPushStack[t] == 0:
                for p in range(mutableStackSize):  # !! factor: numTimesteps * numInstructions * maxScalar
                    stackCarValue[t+1, p].set_to(stackCarValue[t, p])
                    stackCdrValue[t+1, p].set_to(stackCdrValue[t, p])
                stackPtr[t+1].set_to(stackPtr[t])

            # Set instruction pointer for next timestep:
            tmpBranchCond[t].set_to(ScalarIsZero(arg1Val))
            if instruction == numRegInstrs + 0:  # jz
                if tmpBranchCond[t] == 1:
                    instrPtr[t+1].set_to(branchAddr[ip])
                elif tmpBranchCond[t] == 0:
                    instrPtr[t+1].set_to((ip + 1) % programLen)
                isHalted[t+1].set_to(0)
                returnValue[t+1].set_to(0)
            elif instruction == numRegInstrs + 1:  # jnz
                if tmpBranchCond[t] == 1:
                    instrPtr[t+1].set_to((ip + 1) % programLen)
                elif tmpBranchCond[t] == 0:
                    instrPtr[t+1].set_to(branchAddr[ip])
                isHalted[t+1].set_to(0)
                returnValue[t+1].set_to(0)
            elif instruction == numRegInstrs + 2:  # return
                instrPtr[t+1].set_to(ip)
                isHalted[t+1].set_to(1)
                returnValue[t+1].set_to(arg1Val)
            else:
                instrPtr[t+1].set_to((ip + 1) % programLen)
                isHalted[t+1].set_to(0)
                returnValue[t+1].set_to(0)

    elif isHalted[t] == 1:
        for r in range(numRegisters):
            registers[t+1, r].set_to(registers[t, r])
        for p in range(mutableStackSize):
            stackCarValue[t+1, p].set_to(stackCarValue[t, p])
            stackCdrValue[t+1, p].set_to(stackCdrValue[t, p])
        stackPtr[t+1].set_to(stackPtr[t])
        instrPtr[t+1].set_to(instrPtr[t])
        isHalted[t+1].set_to(1)
        returnValue[t+1].set_to(returnValue[t])

# Penalize non-halting programs.
outputTermState.set_to(isHalted[numTimesteps])

outputListCopyPos = Var(maxScalar)[maxScalar + 1]
if expectListOutput == 0:
    # Copy register value to output:
    outputRegVal.set_to(returnValue[numTimesteps])
    # Set list output bits to default values:
    for n in range(maxScalar):
        outputListIsDone[n].set_to(1)
        outputListVal[n].set_to(0)
elif expectListOutput == 1:
    # Set output register value to default:
    outputRegVal.set_to(0)
    # Copy list values out:
    outputListCopyPos[0].set_to(returnValue[numTimesteps])
    for n in range(maxScalar):
        outputListIsDone[n].set_to(ScalarIsZero(outputListCopyPos[n]))
        if outputListIsDone[n] == 1:
            outputListVal[n].set_to(0)
            outputListCopyPos[n + 1].set_to(0)

        elif outputListIsDone[n] == 0:
            with outputListCopyPos[n] as p:
                if p == 0:
                    outputListVal[n].set_to(0)
                    outputListCopyPos[n + 1].set_to(0)
                elif p <= inputStackSize:
                    outputListVal[n].set_to(inputStackCarVal[p - 1])
                    outputListCopyPos[n + 1].set_to(inputStackCdrVal[p - 1])
                else:
                    outputListVal[n].set_to(stackCarValue[numTimesteps, p - inputStackSize - 1])
                    outputListCopyPos[n + 1].set_to(stackCdrValue[numTimesteps, p - inputStackSize - 1])
