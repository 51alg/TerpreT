from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

#### Parameters to the model (changes in this block should not require
#### any changes in the actual model)
inputNum = Hyper()
inputStackSize = Hyper()
prefixLength = Hyper()
lambdaLength = Hyper()
suffixLength = Hyper()

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
else:
    stackPtrAtSuffixStart = stackPtrAtCombinatorStart
stackSize = stackPtrAtSuffixStart + suffixLength
maxScalar = stackSize + 0

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
for i in range(0, prefixLength):
    # We allow access to the inputs and all registers written so far
    prefixInstructionsArg1[i] = Param(inputNum + i)
    prefixInstructionsArg2[i] = Param(inputNum + i)
    prefixInstructionsCondition[i] = Param(inputNum + i)
prefixRegisterNum = inputNum + prefixLength

# Choosing the combinator, its instructions and their arguments:
combinator = Param(numCombinators)
combinatorStartAcc = Param(prefixRegisterNum)
combinatorInputList1 = Param(prefixRegisterNum)
combinatorInputList2 = Param(prefixRegisterNum)

lambdaInstructions = Param(numInstructions)[lambdaLength]
for i in range(0, lambdaLength):
    # We allow access to all inputs, prefix registers, currentElementVal,
    # currentAggregatorVal, index and all registers written so far
    numLambdaInputs = 3
    lambdaInstructionsArg1[i] = Param(prefixRegisterNum + i + numLambdaInputs)
    lambdaInstructionsArg2[i] = Param(prefixRegisterNum + i + numLambdaInputs)
    lambdaInstructionsCondition[i] = Param(prefixRegisterNum + i)
lambdaReturnReg = Param(lambdaLength)

# Suffix instructions and arguments
suffixInstructions = Param(numInstructions)[suffixLength]
for i in range(0, suffixLength):
    # We allow access to the inputs, prefix registers, and the combinator result:
    suffixInstructionsArg1[i] = Param(prefixRegisterNum + 1 + i)
    suffixInstructionsArg2[i] = Param(prefixRegisterNum + 1 + i)
    suffixInstructionsCondition[i] = Param(prefixRegisterNum + 1 + i)

registerNum = prefixRegisterNum + 1 + suffixLength
programReturnReg = Param(registerNum)

#### Execution data description
## Stack
stackCarVal = Var(maxScalar)[stackSize]
stackCdrVal = Var(maxScalar)[stackSize]

## Program registers
regVal = Var(maxScalar)[registerNum]

## Closure registers
# Immutable, but in every combinator timestep, we get fresh ones.
lambdaRegVal = Var(maxScalar)[maxLoopsteps, lambdaLength]

## Aggregator values, one after each loop iteration (+ an initial value)
aggregatorVal = Var(maxScalar)[maxLoopsteps + 1]
curCombinatorElementPtr1 = Var(stackSize)[maxLoopsteps + 1]
curCombinatorElementIntVal1 = Var(maxScalar)[maxLoopsteps + 1]
curCombinatorElementPtr2 = Var(stackSize)[maxLoopsteps + 1]
curCombinatorElementIntVal2 = Var(maxScalar)[maxLoopsteps + 1]

## Temporary things:
# Temp variable that marks that we've reached the end of the list (and
# just sit out the remaining loop steps)
listIsOver = Var(boolSize)[maxLoopsteps + 1]

# Temp variables containing the input arguments (to simplify the execution)
tmpPrefixArg1Val = Var(maxScalar)[prefixLength]
tmpPrefixArg2Val = Var(maxScalar)[prefixLength]
tmpPrefixConditionVal = Var(2)[prefixLength]

tmpLambdaArg1Val = Var(maxScalar)[maxLoopsteps, lambdaLength]
tmpLambdaArg2Val = Var(maxScalar)[maxLoopsteps, lambdaLength]
tmpLambdaConditionVal = Var(2)[maxLoopsteps, lambdaLength]

tmpSuffixArg1Val = Var(maxScalar)[suffixLength]
tmpSuffixArg2Val = Var(maxScalar)[suffixLength]
tmpSuffixConditionVal = Var(2)[suffixLength]

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
    regVal[i].set_to(inputRegVal[i])

# Initialize nil element at bottom of stack:
stackCarVal[0].set_to(0)
stackCdrVal[0].set_to(0)

# Copy input stack into our temporary representation:
for i in range(inputStackSize):
    stackCarVal[1 + i].set_to(inputStackCarVal[i])
    stackCdrVal[1 + i].set_to(inputStackCdrVal[i])

##### Run prefix
for i in range(prefixLength):
    # Aliases for instruction processing. Instructions are
    # of the following form: "out = op arg1 arg2", where
    # arg1 and arg2 are either pointers or integers and
    # are chosen based on the type of the operator.
    outVal = regVal[inputNum + i]
    arg1Val = tmpPrefixArg1Val[i]
    arg2Val = tmpPrefixArg2Val[i]
    condition = tmpPrefixConditionVal[i]

    # Get the inputs:
    with prefixInstructionsArg1[i] as r:
        arg1Val.set_to(regVal[r])
    with prefixInstructionsArg2[i] as r:
        arg2Val.set_to(regVal[r])
    with prefixInstructionsCondition[i] as r:
        condition.set_to(ScalarAsBool(regVal[r]))

    curStackPtr = stackPtrAtPrefixStart + i

    ExecuteInstruction(
        prefixInstructions[i],
        arg1Val, arg2Val, condition, outVal,
        curStackPtr,
        stackCarVal[curStackPtr], stackCdrVal[curStackPtr])

##### If we have no body, do nothing:
if lambdaLength == 0:
    regVal[prefixRegisterNum].set_to(0)

##### Otherwise, set up and run combinator:
else:
    with combinatorInputList1 as combInList1Reg:
        curCombinatorElementPtr1[0].set_to(regVal[combInList1Reg])
    with combinatorInputList2 as combInList2Reg:
        curCombinatorElementPtr2[0].set_to(regVal[combInList2Reg])

    # We initialise the aggregator dependent on the combinator:
    if combinator == 0:  # foldl
        with combinatorStartAcc as combStartAccReg:
            aggregatorVal[0].set_to(regVal[combStartAccReg])
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

            # Execute the body of our lambda:
            for i in range(0, lambdaLength):
                ele1RegisterIdx = prefixRegisterNum + i
                aggRegisterIdx = prefixRegisterNum + i + 1
                idxRegisterIdx = prefixRegisterNum + i + 2

                # Aliases for instruction processing. Instructions are
                # of the following form: "out = op arg1 arg2", where
                # arg1 and arg2 are either pointers or integers and
                # are chosen based on the type of the operator.
                outVal = lambdaRegVal[l, i]
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
                    elif r < prefixRegisterNum:
                        arg1Val.set_to(regVal[r])
                    else:
                        arg1Val.set_to(lambdaRegVal[l, r - prefixRegisterNum])
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
                    elif r < prefixRegisterNum:
                        arg2Val.set_to(regVal[r])
                    else:
                        arg2Val.set_to(lambdaRegVal[l, r - prefixRegisterNum])
                with lambdaInstructionsCondition[i] as r:
                    if r < prefixRegisterNum:
                        condition.set_to(ScalarAsBool(regVal[r]))
                    else:
                        condition.set_to(ScalarAsBool(lambdaRegVal[l, r - prefixRegisterNum]))

                # Stack pointer:
                # number of iterations we already did * size of the closure body
                # + how far we are in this one:
                curStackPtr = stackPtrAtCombinatorStart + l * (lambdaLength + 1) + i

                ExecuteInstruction(
                    lambdaInstructions[i],
                    arg1Val, arg2Val, condition, outVal,
                    curStackPtr,
                    stackCarVal[curStackPtr], stackCdrVal[curStackPtr])

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
                lambdaOut = lambdaRegVal[l, outputRegIndex]

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
                curStackPtr = stackPtrAtCombinatorStart + l * (lambdaLength + 1) + i
                stackCarVal[curStackPtr].set_to(0)
                stackCdrVal[curStackPtr].set_to(0)

    if combinator == 0:  # foldl:
        regVal[prefixRegisterNum].set_to(aggregatorVal[maxLoopsteps])
    elif combinator == 1:  # map:
        # Point to first list element generated in map:
        regVal[prefixRegisterNum].set_to(stackPtrAtCombinatorStart + lambdaLength)
    elif combinator == 2:  # zipWith:
        # Point to first list element generated in zipWith:
        regVal[prefixRegisterNum].set_to(stackPtrAtCombinatorStart + lambdaLength)

##### Run suffix
for i in range(suffixLength):
    # Aliases for instruction processing. Instructions are
    # of the following form: "out = op arg1 arg2", where
    # arg1 and arg2 are either pointers or integers and
    # are chosen based on the type of the operator.
    outVal = regVal[prefixRegisterNum + 1 + i]
    arg1Val = tmpSuffixArg1Val[i]
    arg2Val = tmpSuffixArg2Val[i]
    condition = tmpSuffixConditionVal[i]

    # Get the inputs:
    with suffixInstructionsArg1[i] as r:
        arg1Val.set_to(regVal[r])
    with suffixInstructionsArg2[i] as r:
        arg2Val.set_to(regVal[r])
    with suffixInstructionsCondition[i] as r:
        condition.set_to(ScalarAsBool(regVal[r]))

    curStackPtr = stackPtrAtSuffixStart + i

    ExecuteInstruction(
        suffixInstructions[i],
        arg1Val, arg2Val, condition, outVal,
        curStackPtr,
        stackCarVal[curStackPtr], stackCdrVal[curStackPtr])


outputTermState.set_to(1)
if expectListOutput == 0:
    # Copy register to output:
    with programReturnReg as outputRegIndex:
        outputRegVal.set_to(regVal[outputRegIndex])
    # Set list output bits to default values:
    for n in range(stackSize):
        outputListIsDone[n].set_to(1)
        outputListVal[n].set_to(0)
elif expectListOutput == 1:
    # Set output register value to default:
    outputRegVal.set_to(0)
    # Copy list values out:
    outputListCopyPos = Var(stackSize)[stackSize + 1]
    with programReturnReg as outputRegIndex:
        outputListCopyPos[0].set_to(regVal[outputRegIndex])
    for n in range(stackSize):
        outputListIsDone[n].set_to(PtrIsNull(outputListCopyPos[n], stackSize))
        if outputListIsDone[n] == 1:
            outputListVal[n].set_to(0)
            outputListCopyPos[n + 1].set_to(0)

        elif outputListIsDone[n] == 0:
            with outputListCopyPos[n] as p:
                outputListVal[n].set_to(stackCarVal[p])
                outputListCopyPos[n + 1].set_to(stackCdrVal[p])
