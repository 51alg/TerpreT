from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

#### Parameters to the model (changes in this block should not require
#### any changes in the actual model)
inputNum = Hyper()
inputStackSize = Hyper()
prefixLength = Hyper()
loopBodyLength = Hyper()
suffixLength = Hyper()
extraRegisterNum = Hyper()

#### Inputs:
# We first need to work out the size of the stack.
# The combinator can run at most for the number of input elements + what
# was allocated in the prefix:
maxLoopsteps = inputStackSize + prefixLength
# The number of stack cells is dependent on the number of instructions
# and inputs as follows:
#  - 1 for Nil
#  - inputStackSize
#  - prefixLength (as each instruction can allocate)
#  - maxLoopsteps * lambdaLength (as each lambda instruction can allocate)
#  - suffixLength (as each instruction can allocate)
stackPtrAtPrefixStart = 1 + inputStackSize
stackPtrAtLoopStart = stackPtrAtPrefixStart + prefixLength
numTimesteps = 1 + prefixLength + (loopBodyLength * maxLoopsteps) + suffixLength
stackSize = stackPtrAtLoopStart + maxLoopsteps * loopBodyLength + suffixLength
maxScalar = stackSize + 0

registerNum = inputNum + extraRegisterNum
loopRegisterNum = registerNum + 2

inputRegVal = Input(maxScalar)[inputNum]
inputStackCarVal = Input(maxScalar)[inputStackSize]
inputStackCdrVal = Input(stackSize)[inputStackSize]

#### Outputs:
expectListOutput = Input(2)
outputRegVal = Output(maxScalar)
outputListVal = Output(maxScalar)[stackSize]
outputListIsDone = Output(2)[stackSize]
outputTermState = Output(2)

#### Execution model description
## Loops: foreach in l, foreach in zip l1, l2
numLoops = 2

## Instructions: cons, cdr, car, nil/zero/false, add, inc, eq, gt, and, ite,
##               one/true, noop/copy, dec, or
numInstructions = 14
boolSize = 2
@Runtime([maxScalar, maxScalar], maxScalar)
def Add(x, y): return (x + y) % maxScalar
@Runtime([maxScalar], maxScalar)
def Inc(x): return (x + 1) % maxScalar
@Runtime([maxScalar], maxScalar)
def Dec(x): return (x - 1) % maxScalar  # Python normalizes to 0..maxInt
@Runtime([maxScalar, maxScalar], maxScalar)
def EqTest(a, b): return 1 if a == b else 0
@Runtime([maxScalar, maxScalar], maxScalar)
def GtTest(a, b): return 1 if a > b else 0
@Runtime([maxScalar, maxScalar], maxScalar)
def And(a, b): return 1 if a > 0 and b > 0 else 0
@Runtime([maxScalar, maxScalar], maxScalar)
def Or(a, b): return 1 if a > 0 or b > 0 else 0

# Modeling helper functions, not actual instructions:
@Runtime([registerNum, registerNum], boolSize)
def RegEqTest(a, b): return 1 if a == b else 0
@Runtime([stackSize, stackSize+1], boolSize)
def PtrIsNull(ptr, curStackPtr):
    return 1 if (ptr == 0 or ptr >= curStackPtr) else 0
@Runtime([stackSize, stackSize, stackSize+1], boolSize)
def OnePtrIsNull(ptr1, ptr2, curStackPtr):
    if ptr1 == 0 or ptr1 >= curStackPtr or ptr2 == 0 or ptr2 >= curStackPtr:
        return 1
    else:
        return 0
@Runtime([maxScalar], boolSize)
def ScalarAsBool(x): return 1 if x > 0 else 0

# Prefix instructions and arguments
prefixInstructions = Param(numInstructions)[prefixLength]
prefixInstructionsArg1 = Param(registerNum)[prefixLength]
prefixInstructionsArg2 = Param(registerNum)[prefixLength]
prefixInstructionsCondition = Param(registerNum)[prefixLength]
prefixInstructionsOut = Param(registerNum)[prefixLength]

# Choosing the loop, its instructions and their arguments:
loop = Param(numLoops)
loopInputList1 = Param(registerNum)
loopInputList2 = Param(registerNum)

loopBodyInstructions = Param(numInstructions)[loopBodyLength]
loopBodyInstructionsOut = Param(registerNum)[loopBodyLength]
loopBodyInstructionsArg1 = Param(loopRegisterNum)[loopBodyLength]
loopBodyInstructionsArg2 = Param(loopRegisterNum)[loopBodyLength]
loopBodyInstructionsCondition = Param(registerNum)[loopBodyLength]

# Suffix instructions and arguments
suffixInstructions = Param(numInstructions)[suffixLength]
suffixInstructionsArg1 = Param(registerNum)[suffixLength]
suffixInstructionsArg2 = Param(registerNum)[suffixLength]
suffixInstructionsCondition = Param(registerNum)[suffixLength]
suffixInstructionsOut = Param(registerNum)[suffixLength]

programReturnReg = Param(registerNum)

#### Execution data description
## Stack
stackCarVal = Var(maxScalar)[stackSize]
stackCdrVal = Var(stackSize)[stackSize]

## Program registers
regVal = Var(maxScalar)[numTimesteps, registerNum]

## Pointers to the current loop element, and values:
curLoopElementPtr1 = Var(stackSize)[maxLoopsteps + 1]
curLoopElementPtr2 = Var(stackSize)[maxLoopsteps + 1]
curLoopElementVal1 = Var(maxScalar)[maxLoopsteps]
curLoopElementVal2 = Var(maxScalar)[maxLoopsteps]

## Temporary things:
# Temp variable that marks that we've reached the end of the list (and
# just sit out the remaining loop steps)
listIsOver = Var(boolSize)[maxLoopsteps + 1]

# Temp variables containing the input arguments (to simplify the execution)
tmpPrefixArg1Val = Var(maxScalar)[prefixLength]
tmpPrefixArg2Val = Var(maxScalar)[prefixLength]
tmpPrefixOutVal = Var(maxScalar)[prefixLength]
tmpPrefixConditionVal = Var(2)[prefixLength]
tmpPrefixDoWriteReg = Var(2)[prefixLength, registerNum]

tmpLoopBodyArg1Val = Var(maxScalar)[maxLoopsteps, loopBodyLength]
tmpLoopBodyArg2Val = Var(maxScalar)[maxLoopsteps, loopBodyLength]
tmpLoopBodyOutVal = Var(maxScalar)[maxLoopsteps, loopBodyLength]
tmpLoopBodyConditionVal = Var(2)[maxLoopsteps, loopBodyLength]
tmpLoopBodyDoWriteReg = Var(2)[maxLoopsteps, loopBodyLength, registerNum]

tmpSuffixArg1Val = Var(maxScalar)[suffixLength]
tmpSuffixArg2Val = Var(maxScalar)[suffixLength]
tmpSuffixOutVal = Var(maxScalar)[suffixLength]
tmpSuffixConditionVal = Var(2)[suffixLength]
tmpSuffixDoWriteReg = Var(2)[suffixLength, registerNum]

@Inline()
def ExecuteInstruction(instruction,
                       arg1Val, arg2Val, condition, outVal,
                       curStackPtr, outCarStack, outCdrStack):
    # Do the actual execution. Every instruction sets its
    # corresponding register value, and the two heap cells:
    if instruction == 0:  # cons
        outVal.set_to(curStackPtr)
        outCarStack.set_to(arg1Val)
        outCdrStack.set_to(arg2Val)
    elif instruction == 1:  # car
        with arg1Val as p:
            if p < curStackPtr:
                outVal.set_to(stackCarVal[p])
            else:
                outVal.set_to(0)
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 2:  # cdr
        with arg1Val as p:
            if p < curStackPtr:
                outVal.set_to(stackCdrVal[p])
            else:
                outVal.set_to(0)
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 3:  # nil/zero/false
        outVal.set_to(0)
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 4:  # add
        outVal.set_to(Add(arg1Val, arg2Val))
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 5:  # inc
        outVal.set_to(Inc(arg1Val))
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 6:  # eq
        outVal.set_to(EqTest(arg1Val, arg2Val))
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 7:  # gt
        outVal.set_to(GtTest(arg1Val, arg2Val))
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 8:  # and
        outVal.set_to(And(arg1Val, arg2Val))
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 9:  # ite
        if condition == 1:
            outVal.set_to(arg1Val)
        elif condition == 0:
            outVal.set_to(arg2Val)
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 10:  # one/true
        outVal.set_to(1)
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 11:  # noop/copy
        outVal.set_to(arg1Val)
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 12:  # dec
        outVal.set_to(Dec(arg1Val))
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)
    elif instruction == 13:  # or
        outVal.set_to(Or(arg1Val, arg2Val))
        # These just stay empty:
        outCarStack.set_to(0)
        outCdrStack.set_to(0)

##### Setting up inputs:
# Copy input registers to temporary registers:
for i in range(inputNum):
    regVal[0, i].set_to(inputRegVal[i])
for r in range(inputNum, registerNum):
    regVal[0, r].set_to(0)

# Initialize nil element at bottom of stack:
stackCarVal[0].set_to(0)
stackCdrVal[0].set_to(0)

# Copy input stack into our temporary representation:
for i in range(inputStackSize):
    stackCarVal[1 + i].set_to(inputStackCarVal[i])
    stackCdrVal[1 + i].set_to(inputStackCdrVal[i])

##### Run prefix
for t in range(prefixLength):
    # Aliases for instruction processing. Instructions are
    # of the following form: "out = op arg1 arg2", where
    # arg1 and arg2 are either pointers or integers and
    # are chosen based on the type of the operator.
    outVal = tmpPrefixOutVal[t]
    arg1Val = tmpPrefixArg1Val[t]
    arg2Val = tmpPrefixArg2Val[t]
    conditionVal = tmpPrefixConditionVal[t]

    # Get the inputs:
    with prefixInstructionsArg1[t] as r:
        arg1Val.set_to(regVal[t, r])
    with prefixInstructionsArg2[t] as r:
        arg2Val.set_to(regVal[t, r])
    with prefixInstructionsCondition[t] as r:
        conditionVal.set_to(ScalarAsBool(regVal[t, r]))

    curStackPtr = stackPtrAtPrefixStart + t

    ExecuteInstruction(
        prefixInstructions[t],
        arg1Val, arg2Val, conditionVal, outVal,
        curStackPtr,
        stackCarVal[curStackPtr], stackCdrVal[curStackPtr])

    for r in range(registerNum):
        tmpPrefixDoWriteReg[t, r].set_to(RegEqTest(prefixInstructionsOut[t], r))
        if tmpPrefixDoWriteReg[t, r] == 0:
            regVal[t + 1, r].set_to(regVal[t, r])
        elif tmpPrefixDoWriteReg[t, r] == 1:
            regVal[t + 1, r].set_to(outVal)

t = prefixLength
##### Set up and run loop:
with loopInputList1 as loopList1Reg:
    curLoopElementPtr1[0].set_to(regVal[t, loopList1Reg])
with loopInputList2 as loopList2Reg:
    curLoopElementPtr2[0].set_to(regVal[t, loopList2Reg])

#Check if we are done with the list:
if loop == 0:  # foreach
    listIsOver[0].set_to(PtrIsNull(curLoopElementPtr1[0],
                                   stackPtrAtLoopStart))
elif loop == 1:  # foreach zip
    listIsOver[0].set_to(OnePtrIsNull(curLoopElementPtr1[0],
                                      curLoopElementPtr2[0],
                                      stackPtrAtLoopStart))

ele1RegisterIdx = registerNum
ele2RegisterIdx = registerNum + 1
for l in range(maxLoopsteps):
    t = prefixLength + l * loopBodyLength
    # Extract current list elements and already compute next element pointer:
    if listIsOver[l] == 0:
        with curLoopElementPtr1[l] as curPtr1:
            if curPtr1 < stackPtrAtLoopStart:
                curLoopElementVal1[l].set_to(stackCarVal[curPtr1])
                curLoopElementPtr1[l + 1].set_to(stackCdrVal[curPtr1])
            else:
                curLoopElementVal1[l].set_to(0)
                curLoopElementPtr1[l + 1].set_to(0)
        with curLoopElementPtr2[l] as curPtr2:
            if curPtr2 < stackPtrAtLoopStart + l * loopBodyLength:
                curLoopElementVal2[l].set_to(stackCarVal[curPtr2])
                curLoopElementPtr2[l + 1].set_to(stackCdrVal[curPtr2])
            else:
                curLoopElementVal2[l].set_to(0)
                curLoopElementPtr2[l + 1].set_to(0)

        # Execute the body of our loopBody:
        for i in range(0, loopBodyLength):
            t = prefixLength + l * loopBodyLength + i

            # Aliases for instruction processing. Instructions are
            # of the following form: "out = op arg1 arg2", where
            # arg1 and arg2 are either pointers or integers and
            # are chosen based on the type of the operator.
            outVal = tmpLoopBodyOutVal[l, i]
            arg1Val = tmpLoopBodyArg1Val[l, i]
            arg2Val = tmpLoopBodyArg2Val[l, i]
            conditionVal = tmpLoopBodyConditionVal[l, i]

            # Get the inputs:
            with loopBodyInstructionsArg1[i] as r:
                if r == ele1RegisterIdx:
                    arg1Val.set_to(curLoopElementVal1[l])
                elif r == ele2RegisterIdx:
                    arg1Val.set_to(curLoopElementVal2[l])
                else:
                    arg1Val.set_to(regVal[t, r])
            with loopBodyInstructionsArg2[i] as r:
                if r == ele1RegisterIdx:
                    arg2Val.set_to(curLoopElementVal1[l])
                elif r == ele2RegisterIdx:
                    arg2Val.set_to(curLoopElementVal2[l])
                else:
                    arg2Val.set_to(regVal[t, r])
            with loopBodyInstructionsCondition[i] as r:
                conditionVal.set_to(ScalarAsBool(regVal[t, r]))

            # Stack pointer:
            # number of iterations we already did * size of the closure body
            # + how far we are in this one:
            curStackPtr = stackPtrAtLoopStart + l * loopBodyLength + i

            ExecuteInstruction(
                loopBodyInstructions[i],
                arg1Val, arg2Val, conditionVal, outVal,
                curStackPtr,
                stackCarVal[curStackPtr], stackCdrVal[curStackPtr])

            for r in range(registerNum):
                tmpLoopBodyDoWriteReg[l, i, r].set_to(RegEqTest(loopBodyInstructionsOut[i], r))
                if tmpLoopBodyDoWriteReg[l, i, r] == 0:
                    regVal[t + 1, r].set_to(regVal[t, r])
                elif tmpLoopBodyDoWriteReg[l, i, r] == 1:
                    regVal[t + 1, r].set_to(outVal)

        # Check if the next list element is empty:
        if loop == 0:  # foreach
            listIsOver[l + 1].set_to(PtrIsNull(curLoopElementPtr1[l + 1],
                                               stackPtrAtLoopStart))
        elif loop == 1:  # foreach zip
            listIsOver[l + 1].set_to(OnePtrIsNull(curLoopElementPtr1[l + 1],
                                                  curLoopElementPtr2[l + 1],
                                                  stackPtrAtLoopStart))

    elif listIsOver[l] == 1:
        listIsOver[l + 1].set_to(1)
        curLoopElementPtr1[l + 1].set_to(0)
        curLoopElementPtr2[l + 1].set_to(0)

        # We still need to initialise the stack cells for all these steps to 0:
        for i in range(0, loopBodyLength):
            # Copy register forwards.
            t = prefixLength + l * loopBodyLength + i
            for r in range(registerNum):
                regVal[t + 1, r].set_to(regVal[t, r])

            curStackPtr = stackPtrAtLoopStart + l * loopBodyLength + i
            stackCarVal[curStackPtr].set_to(0)
            stackCdrVal[curStackPtr].set_to(0)

##### Run suffix
stackPtrAtSuffixStart = stackPtrAtLoopStart + maxLoopsteps * loopBodyLength
for i in range(suffixLength):
    t = prefixLength + loopBodyLength * maxLoopsteps + i

    # Aliases for instruction processing. Instructions are
    # of the following form: "out = op arg1 arg2", where
    # arg1 and arg2 are either pointers or integers and
    # are chosen based on the type of the operator.
    outVal = tmpSuffixOutVal[i]
    arg1Val = tmpSuffixArg1Val[i]
    arg2Val = tmpSuffixArg2Val[i]
    conditionVal = tmpSuffixConditionVal[i]

    # Get the inputs:
    with suffixInstructionsArg1[i] as r:
        arg1Val.set_to(regVal[t, r])
    with suffixInstructionsArg2[i] as r:
        arg2Val.set_to(regVal[t, r])
    with suffixInstructionsCondition[i] as r:
        conditionVal.set_to(ScalarAsBool(regVal[t, r]))

    curStackPtr = stackPtrAtSuffixStart + i

    ExecuteInstruction(
        suffixInstructions[i],
        arg1Val, arg2Val, conditionVal, outVal,
        curStackPtr,
        stackCarVal[curStackPtr], stackCdrVal[curStackPtr])

    for r in range(registerNum):
        tmpSuffixDoWriteReg[i, r].set_to(RegEqTest(suffixInstructionsOut[i], r))
        if tmpSuffixDoWriteReg[i, r] == 0:
            regVal[t + 1, r].set_to(regVal[t, r])
        elif tmpSuffixDoWriteReg[i, r] == 1:
            regVal[t + 1, r].set_to(outVal)


outputTermState.set_to(1)
outputListCopyPos = Var(stackSize)[stackSize + 1]
if expectListOutput == 0:
    # Copy register to output:
    with programReturnReg as outputRegIndex:
        outputRegVal.set_to(regVal[numTimesteps - 1, outputRegIndex])
    # Set list output bits to default values:
    for n in range(stackSize):
        outputListIsDone[n].set_to(1)
        outputListVal[n].set_to(0)
elif expectListOutput == 1:
    # Set output register value to default:
    outputRegVal.set_to(0)
    # Copy list values out:
    with programReturnReg as outputRegIndex:
        outputListCopyPos[0].set_to(regVal[numTimesteps - 1, outputRegIndex])
    for n in range(stackSize):
        outputListIsDone[n].set_to(PtrIsNull(outputListCopyPos[n], stackSize))
        if outputListIsDone[n] == 1:
            outputListVal[n].set_to(0)
            outputListCopyPos[n + 1].set_to(0)

        elif outputListIsDone[n] == 0:
            with outputListCopyPos[n] as p:
                outputListVal[n].set_to(stackCarVal[p])
                outputListCopyPos[n + 1].set_to(stackCdrVal[p])
