from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

#### Parameters to the model (changes in this block should not require
#### any changes in the actual model)
inputNum = Hyper()
inputStackSize = Hyper()
prefixLength = Hyper()
lambdaLength = Hyper()
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
#  - maxLoopsteps (as we create a list element for each map/zipWith iteration)
#  - maxLoopsteps * lambdaLength (as each lambda instruction can allocate)
#  - suffixLength (as each instruction can allocate)
stackPtrAtPrefixStart = 1 + inputStackSize
stackPtrAtCombinatorStart = stackPtrAtPrefixStart + prefixLength
if lambdaLength > 0:
    stackPtrAtSuffixStart = stackPtrAtCombinatorStart + maxLoopsteps * (1 + lambdaLength)
    numTimesteps = prefixLength + ((lambdaLength + 1) * maxLoopsteps) + suffixLength + 2
else:
    stackPtrAtSuffixStart = stackPtrAtCombinatorStart
    numTimesteps = prefixLength + suffixLength + 2
stackSize = stackPtrAtSuffixStart + suffixLength
maxScalar = stackSize + 0

registerNum = inputNum + extraRegisterNum
lambdaRegisterNum = registerNum + 3

inputRegVal = Input(maxScalar)[inputNum]
inputStackCarVal = Input(maxScalar)[inputStackSize]
inputStackCdrVal = Input(stackSize)[inputStackSize]

#### Outputs:
outputRegVal = Output(maxScalar)
outputListVal = Output(maxScalar)[stackSize]
outputTermState = Output(2)

#### Execution model description
## Combinators: foldl, map, zipWith
numCombinators = 3

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

# Choosing the combinator, its instructions and their arguments:
combinator = Param(numCombinators)
combinatorStartAcc = Param(registerNum)
combinatorInputList1 = Param(registerNum)
combinatorInputList2 = Param(registerNum)
combinatorOut = Param(registerNum)

lambdaInstructions = Param(numInstructions)[lambdaLength]
lambdaInstructionsOut = Param(registerNum)[lambdaLength]
lambdaInstructionsArg1 = Param(lambdaRegisterNum)[lambdaLength]
lambdaInstructionsArg2 = Param(lambdaRegisterNum)[lambdaLength]
lambdaInstructionsCondition = Param(registerNum)[lambdaLength]
lambdaReturnReg = Param(registerNum)

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

## Aggregator values, one after each loop iteration (+ an initial value)
aggregatorVal = Var(maxScalar)[maxLoopsteps + 1]
curCombinatorElementPtr1 = Var(maxScalar)[maxLoopsteps + 1]
curCombinatorElementIntVal1 = Var(maxScalar)[maxLoopsteps + 1]
curCombinatorElementPtr2 = Var(maxScalar)[maxLoopsteps + 1]
curCombinatorElementIntVal2 = Var(maxScalar)[maxLoopsteps + 1]

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

tmpLambdaArg1Val = Var(maxScalar)[maxLoopsteps, lambdaLength]
tmpLambdaArg2Val = Var(maxScalar)[maxLoopsteps, lambdaLength]
tmpLambdaOutVal = Var(maxScalar)[maxLoopsteps, lambdaLength]
tmpLambdaConditionVal = Var(2)[maxLoopsteps, lambdaLength]
tmpLambdaDoWriteReg = Var(2)[maxLoopsteps, lambdaLength, registerNum]

tmpOutputDoWriteReg = Var(2)[registerNum]

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
##### If we have no body, do nothing:
if lambdaLength == 0:
    for r in range(registerNum):
        tmpOutputDoWriteReg[r].set_to(RegEqTest(combinatorOut, r))
        if tmpOutputDoWriteReg[r] == 0:
            regVal[t + 1, r].set_to(regVal[t, r])
        elif tmpOutputDoWriteReg[r] == 1:
            regVal[t + 1, r].set_to(0)

##### Otherwise, set up and run combinator:
else:
    ele1RegisterIdx = registerNum
    aggRegisterIdx = registerNum + 1
    idxRegisterIdx = registerNum + 2

    with combinatorInputList1 as combInList1Reg:
        curCombinatorElementPtr1[0].set_to(regVal[t, combInList1Reg])
    with combinatorInputList2 as combInList2Reg:
        curCombinatorElementPtr2[0].set_to(regVal[t, combInList2Reg])

    # We initialise the aggregator dependent on the combinator:
    if combinator == 0:  # foldl
        with combinatorStartAcc as combStartAccReg:
            aggregatorVal[0].set_to(regVal[t, combStartAccReg])
        listIsOver[0].set_to(PtrIsNull(curCombinatorElementPtr1[0],
                                       stackPtrAtCombinatorStart))
    elif combinator == 1:  # map
        aggregatorVal[0].set_to(0)
        listIsOver[0].set_to(PtrIsNull(curCombinatorElementPtr1[0],
                                       stackPtrAtCombinatorStart))
    elif combinator == 2:  # zipWith
        aggregatorVal[0].set_to(0)
        listIsOver[0].set_to(OnePtrIsNull(curCombinatorElementPtr1[0],
                                          curCombinatorElementPtr2[0],
                                          stackPtrAtCombinatorStart))

    for l in range(maxLoopsteps):
        t = prefixLength + ((lambdaLength + 1) * l)

        if listIsOver[l] == 0:
            # Extract current list elements and already compute next element pointer:
            with curCombinatorElementPtr1[l] as curPtr1:
                if curPtr1 < stackPtrAtCombinatorStart:
                    curCombinatorElementIntVal1[l].set_to(stackCarVal[curPtr1])
                    curCombinatorElementPtr1[l + 1].set_to(stackCdrVal[curPtr1])
                else:
                    curCombinatorElementIntVal1[l].set_to(0)
                    curCombinatorElementPtr1[l + 1].set_to(0)
            with curCombinatorElementPtr2[l] as curPtr2:
                if curPtr2 < stackPtrAtCombinatorStart:
                    curCombinatorElementIntVal2[l].set_to(stackCarVal[curPtr2])
                    curCombinatorElementPtr2[l + 1].set_to(stackCdrVal[curPtr2])
                else:
                    curCombinatorElementIntVal2[l].set_to(0)
                    curCombinatorElementPtr2[l + 1].set_to(0)

            # Set up local registers:
            for r in range(inputNum):
                regVal[t + 1, r].set_to(regVal[t, r])
            for r in range(inputNum, registerNum):
                regVal[t + 1, r].set_to(0)

            # Execute the body of our lambda:
            for i in range(0, lambdaLength):
                t = prefixLength + ((lambdaLength + 1) * l) + 1 + i

                # Aliases for instruction processing. Instructions are
                # of the following form: "out = op arg1 arg2", where
                # arg1 and arg2 are either pointers or integers and
                # are chosen based on the type of the operator.
                outVal = tmpLambdaOutVal[l, i]
                arg1Val = tmpLambdaArg1Val[l, i]
                arg2Val = tmpLambdaArg2Val[l, i]
                condition = tmpLambdaConditionVal[l, i]

                # Get the inputs:
                with lambdaInstructionsArg1[i] as r:
                    if r == ele1RegisterIdx:
                        arg1Val.set_to(curCombinatorElementIntVal1[l])
                    elif r == aggRegisterIdx:
                        if combinator == 0:  # foldl
                            arg1Val.set_to(aggregatorVal[l])
                        elif combinator == 1:  # map
                            arg1Val.set_to(0)
                        elif combinator == 2:  # zipWith
                            arg1Val.set_to(curCombinatorElementIntVal2[l])
                    elif r == idxRegisterIdx:
                        arg1Val.set_to(l)
                    else:
                        arg1Val.set_to(regVal[t, r])
                with lambdaInstructionsArg2[i] as r:
                    if r == ele1RegisterIdx:
                        arg2Val.set_to(curCombinatorElementIntVal1[l])
                    elif r == aggRegisterIdx:
                        if combinator == 0:  # foldl
                            arg2Val.set_to(aggregatorVal[l])
                        elif combinator == 1:  # map
                            arg2Val.set_to(0)
                        elif combinator == 2:  # zipWith
                            arg2Val.set_to(curCombinatorElementIntVal2[l])
                    elif r == idxRegisterIdx:
                        arg2Val.set_to(l)
                    else:
                        arg2Val.set_to(regVal[t, r])
                with lambdaInstructionsCondition[i] as r:
                    condition.set_to(ScalarAsBool(regVal[t, r]))

                # Stack pointer:
                # number of iterations we already did * size of the closure body
                # + how far we are in this one:
                curStackPtr = stackPtrAtCombinatorStart + l * (lambdaLength + 1) + i

                ExecuteInstruction(
                    lambdaInstructions[i],
                    arg1Val, arg2Val, condition, outVal,
                    curStackPtr,
                    stackCarVal[curStackPtr], stackCdrVal[curStackPtr])

                for r in range(registerNum):
                    tmpLambdaDoWriteReg[l, i, r].set_to(RegEqTest(lambdaInstructionsOut[i], r))
                    if tmpLambdaDoWriteReg[l, i, r] == 0:
                        regVal[t + 1, r].set_to(regVal[t, r])
                    elif tmpLambdaDoWriteReg[l, i, r] == 1:
                        regVal[t + 1, r].set_to(outVal)

            t = prefixLength + ((lambdaLength + 1) * l) + lambdaLength + 1

            # Check if the next list element is empty:
            if combinator == 0:  # foldl
                listIsOver[l + 1].set_to(PtrIsNull(curCombinatorElementPtr1[l + 1],
                                                   stackPtrAtCombinatorStart))
            elif combinator == 1:  # map
                listIsOver[l + 1].set_to(PtrIsNull(curCombinatorElementPtr1[l + 1],
                                                   stackPtrAtCombinatorStart))
            elif combinator == 2:  # zipWith
                listIsOver[l + 1].set_to(OnePtrIsNull(curCombinatorElementPtr1[l + 1],
                                                      curCombinatorElementPtr2[l + 1],
                                                      stackPtrAtCombinatorStart))

            # Update aggregator:
            newStackPtr = stackPtrAtCombinatorStart + l * (lambdaLength + 1) + lambdaLength
            nextGeneratedStackPtr = stackPtrAtCombinatorStart + (l + 1) * (lambdaLength + 1) + lambdaLength

            with lambdaReturnReg as outputRegIndex:
                lambdaOut = regVal[t, outputRegIndex]

                if combinator == 0:  # foldl
                    aggregatorVal[l + 1].set_to(lambdaOut)
                    # These just stay empty:
                    stackCarVal[newStackPtr].set_to(0)
                    stackCdrVal[newStackPtr].set_to(0)
                elif combinator == 1:  # map
                    # We create a new stack element from the result register,
                    # and point to the next element we will create:
                    aggregatorVal[l + 1].set_to(0)
                    stackCarVal[newStackPtr].set_to(lambdaOut)
                    if listIsOver[l + 1] == 1:
                        stackCdrVal[newStackPtr].set_to(0)
                    elif listIsOver[l + 1] == 0:
                        if l + 1 < maxLoopsteps:
                            stackCdrVal[newStackPtr].set_to(nextGeneratedStackPtr)
                        else:
                            stackCdrVal[newStackPtr].set_to(0)
                elif combinator == 2:  # zipWith
                    # We create a new stack element from the result register,
                    # and point to the next element we will create:
                    aggregatorVal[l + 1].set_to(0)
                    stackCarVal[newStackPtr].set_to(lambdaOut)
                    if listIsOver[l + 1] == 1:
                        stackCdrVal[newStackPtr].set_to(0)
                    elif listIsOver[l + 1] == 0:
                        if l + 1 < maxLoopsteps:
                            stackCdrVal[newStackPtr].set_to(nextGeneratedStackPtr)
                        else:
                            stackCdrVal[newStackPtr].set_to(0)

        elif listIsOver[l] == 1:
            listIsOver[l + 1].set_to(1)
            aggregatorVal[l + 1].set_to(aggregatorVal[l])
            curCombinatorElementPtr1[l + 1].set_to(0)
            curCombinatorElementPtr2[l + 1].set_to(0)

            # We still need to initialise the stack cells for all these steps to 0:
            for i in range(0, lambdaLength + 1):
                # Copy register forwards.
                t = prefixLength + ((lambdaLength + 1) * l) + i
                for r in range(registerNum):
                    regVal[t + 1, r].set_to(regVal[t, r])

                curStackPtr = stackPtrAtCombinatorStart + l * (lambdaLength + 1) + i
                stackCarVal[curStackPtr].set_to(0)
                stackCdrVal[curStackPtr].set_to(0)

    t = prefixLength + ((lambdaLength + 1) * maxLoopsteps)
    for r in range(registerNum):
        tmpOutputDoWriteReg[r].set_to(RegEqTest(combinatorOut, r))
        if tmpOutputDoWriteReg[r] == 0:
            regVal[t + 1, r].set_to(regVal[prefixLength, r])
        elif tmpOutputDoWriteReg[r] == 1:
            if combinator == 0:  # foldl
                regVal[t+1, r].set_to(aggregatorVal[maxLoopsteps])
            elif combinator == 1:  # map:
                # Point to first list element generated in map:
                regVal[t+1, r].set_to(stackPtrAtCombinatorStart + lambdaLength)
            elif combinator == 2:  # zipWith:
                # Point to first list element generated in zipWith:
                regVal[t+1, r].set_to(stackPtrAtCombinatorStart + lambdaLength)

##### Run suffix
for i in range(suffixLength):
    if lambdaLength > 0:
        t = prefixLength + ((lambdaLength + 1) * maxLoopsteps) + 1 + i
    else:
        t = prefixLength + 1 + i

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
            regVal[t + 1,r].set_to(regVal[t, r])
        elif tmpSuffixDoWriteReg[i, r] == 1:
            regVal[t + 1,r].set_to(outVal)


outputTermState.set_to(1)

# Copy register to output:
with programReturnReg as outputRegIndex:
    outputRegVal.set_to(regVal[numTimesteps - 1, outputRegIndex])

# Copy list values out:
outputListCopyPos = Var(stackSize)[stackSize + 1]
outputListCopyPos[0].set_to(outputRegVal)
for n in range(stackSize):
    with outputListCopyPos[n] as p:
        outputListVal[n].set_to(stackCarVal[p])
        outputListCopyPos[n + 1].set_to(stackCdrVal[p])
