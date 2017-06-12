from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

#### Parameters to the model (changes in this block should not require
#### any changes in the actual model)
maxInt = Hyper()
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

inputRegIntVal = Input(maxInt)[inputNum]
inputRegPtrVal = Input(stackSize)[inputNum]
inputRegBoolVal = Input(2)[inputNum]
inputStackCarVal = Input(maxInt)[inputStackSize]
inputStackCdrVal = Input(stackSize)[inputStackSize]

#### Outputs:
expectListOutput = Input(2)
outputRegIntVal = Output(maxInt)
outputRegBoolVal = Output(2)
outputListVal = Output(maxInt)[stackSize]
outputListIsDone = Output(2)[stackSize]
outputTermState = Output(2)

#### Execution model description
## Combinators: foldl, map, zipWith
numCombinators = 3

## Instructions: cons, cdr, car, nil/zero/false, add, inc, eq, gt, and, ite,
##               one/true, noop/copy, dec, or
numInstructions = 14
boolSize = 2
@Runtime([maxInt, maxInt], maxInt)
def Add(x, y): return (x + y) % maxInt
@Runtime([maxInt], maxInt)
def Inc(x): return (x + 1) % maxInt
@Runtime([maxInt], maxInt)
def Dec(x): return (x - 1) % maxInt  # Python normalizes to 0..maxInt
@Runtime([maxInt, maxInt], boolSize)
def EqTest(a, b): return 1 if a == b else 0
@Runtime([maxInt, maxInt], boolSize)
def GtTest(a, b): return 1 if a > b else 0
@Runtime([boolSize, boolSize], boolSize)
def And(a, b): return 1 if a > 0 and b > 0 else 0
@Runtime([boolSize, boolSize], boolSize)
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
stackCarVal = Var(maxInt)[stackSize]
stackCdrVal = Var(stackSize)[stackSize]

## Program registers
regIntVal = Var(maxInt)[registerNum]
regPtrVal = Var(stackSize)[registerNum]
regBoolVal = Var(2)[registerNum]

## Closure registers
# Immutable, but in every combinator timestep, we get fresh ones.
lambdaRegIntVal = Var(maxInt)[maxLoopsteps, lambdaLength]
lambdaRegPtrVal = Var(stackSize)[maxLoopsteps, lambdaLength]
lambdaRegBoolVal = Var(2)[maxLoopsteps, lambdaLength]

## State for the combinator
# Aggregator and list element values for each iteration:
aggregatorIntVal = Var(maxInt)[maxLoopsteps + 1]
aggregatorPtrVal = Var(stackSize)[maxLoopsteps + 1]
aggregatorBoolVal = Var(2)[maxLoopsteps + 1]
curCombinatorElementPtr1 = Var(stackSize)[maxLoopsteps + 1]
curCombinatorElementIntVal1 = Var(maxInt)[maxLoopsteps + 1]
curCombinatorElementPtr2 = Var(stackSize)[maxLoopsteps + 1]
curCombinatorElementIntVal2 = Var(maxInt)[maxLoopsteps + 1]

## Temporary things:
# Temp variable that marks that we've reached the end of the list (and
# just sit out the remaining loop steps)
listIsOver = Var(boolSize)[maxLoopsteps + 1]

# Temp variables containing input arguments (to simplify the execution)
tmpPrefixArg1IntVal = Var(maxInt)[prefixLength]
tmpPrefixArg1PtrVal = Var(stackSize)[prefixLength]
tmpPrefixArg1BoolVal = Var(2)[prefixLength]
tmpPrefixArg2IntVal = Var(maxInt)[prefixLength]
tmpPrefixArg2PtrVal = Var(stackSize)[prefixLength]
tmpPrefixArg2BoolVal = Var(2)[prefixLength]
tmpPrefixConditionVal = Var(2)[prefixLength]

tmpLambdaArg1IntVal = Var(maxInt)[maxLoopsteps, lambdaLength]
tmpLambdaArg1PtrVal = Var(stackSize)[maxLoopsteps, lambdaLength]
tmpLambdaArg1BoolVal = Var(2)[maxLoopsteps, lambdaLength]
tmpLambdaArg2IntVal = Var(maxInt)[maxLoopsteps, lambdaLength]
tmpLambdaArg2PtrVal = Var(stackSize)[maxLoopsteps, lambdaLength]
tmpLambdaArg2BoolVal = Var(2)[maxLoopsteps, lambdaLength]
tmpLambdaConditionVal = Var(2)[maxLoopsteps, lambdaLength]

tmpSuffixArg1IntVal = Var(maxInt)[suffixLength]
tmpSuffixArg1PtrVal = Var(stackSize)[suffixLength]
tmpSuffixArg1BoolVal = Var(2)[suffixLength]
tmpSuffixArg2IntVal = Var(maxInt)[suffixLength]
tmpSuffixArg2PtrVal = Var(stackSize)[suffixLength]
tmpSuffixArg2BoolVal = Var(2)[suffixLength]
tmpSuffixConditionVal = Var(2)[suffixLength]

@Inline()
def ExecuteInstruction(instruction,
                       arg1Ptr, arg1Int, arg1Bool,
                       arg2Ptr, arg2Int, arg2Bool,
                       condition,
                       outPtr, outInt, outBool,
                       curStackPtr, outIntStack, outPtrStack):
    # Do the actual execution. Every instruction sets its
    # corresponding register value, and the two heap cells:
    if instruction == 0:  # cons
        outInt.set_to(0)
        outPtr.set_to(curStackPtr)
        outBool.set_to(0)
        outIntStack.set_to(arg1Int)
        outPtrStack.set_to(arg2Ptr)
    elif instruction == 1:  # car
        with arg1Ptr as p:
            if p < curStackPtr:
                outInt.set_to(stackCarVal[p])
            else:
                outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(0)
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 2:  # cdr
        outInt.set_to(0)
        with arg1Ptr as p:
            if p < curStackPtr:
                outPtr.set_to(stackCdrVal[p])
            else:
                outPtr.set_to(0)
        outBool.set_to(0)
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 3:  # nil/zero/false
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(0)
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 4:  # add
        outInt.set_to(Add(arg1Int, arg2Int))
        outPtr.set_to(0)
        outBool.set_to(0)
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 5:  # inc
        outInt.set_to(Inc(arg1Int))
        outPtr.set_to(0)
        outBool.set_to(0)
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 6:  # eq
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(EqTest(arg1Int, arg2Int))
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 7:  # gt
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(GtTest(arg1Int, arg2Int))
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 8:  # and
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(And(arg1Bool, arg2Bool))
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 9:  # ite
        if condition == 1:
            outInt.set_to(arg1Int)
            outPtr.set_to(arg1Ptr)
            outBool.set_to(arg1Bool)
        elif condition == 0:
            outInt.set_to(arg2Int)
            outPtr.set_to(arg2Ptr)
            outBool.set_to(arg2Bool)
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 10:  # one/true
        outInt.set_to(1)
        outPtr.set_to(0)
        outBool.set_to(1)
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 11:  # noop/copy
        outInt.set_to(arg1Int)
        outPtr.set_to(arg1Ptr)
        outBool.set_to(arg1Bool)
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 12:  # dec
        outInt.set_to(Dec(arg1Int))
        outPtr.set_to(0)
        outBool.set_to(0)
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 13:  # or
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(Or(arg1Bool, arg2Bool))
        # These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)

##### Setting up inputs:
# Copy input registers to temporary registers:
for i in range(inputNum):
    regIntVal[i].set_to(inputRegIntVal[i])
    regPtrVal[i].set_to(inputRegPtrVal[i])
    regBoolVal[i].set_to(inputRegBoolVal[i])

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
    outInt = regIntVal[inputNum + i]
    outPtr = regPtrVal[inputNum + i]
    outBool = regBoolVal[inputNum + i]
    arg1Int = tmpPrefixArg1IntVal[i]
    arg1Ptr = tmpPrefixArg1PtrVal[i]
    arg1Bool = tmpPrefixArg1BoolVal[i]
    arg2Int = tmpPrefixArg2IntVal[i]
    arg2Ptr = tmpPrefixArg2PtrVal[i]
    arg2Bool = tmpPrefixArg2BoolVal[i]
    condition = tmpPrefixConditionVal[i]

    # Get the inputs:
    with prefixInstructionsArg1[i] as r:
        arg1Int.set_to(regIntVal[r])
        arg1Ptr.set_to(regPtrVal[r])
        arg1Bool.set_to(regBoolVal[r])
    with prefixInstructionsArg2[i] as r:
        arg2Int.set_to(regIntVal[r])
        arg2Ptr.set_to(regPtrVal[r])
        arg2Bool.set_to(regBoolVal[r])
    with prefixInstructionsCondition[i] as r:
        condition.set_to(regBoolVal[r])

    curStackPtr = stackPtrAtPrefixStart + i

    ExecuteInstruction(
        prefixInstructions[i],
        arg1Ptr, arg1Int, arg1Bool,
        arg2Ptr, arg2Int, arg2Bool,
        condition,
        outPtr, outInt, outBool,
        curStackPtr,
        stackCarVal[curStackPtr], stackCdrVal[curStackPtr])

##### If we have no body, do nothing:
if lambdaLength == 0:
    regIntVal[prefixRegisterNum].set_to(0)
    regPtrVal[prefixRegisterNum].set_to(0)
    regBoolVal[prefixRegisterNum].set_to(0)

##### Otherwise, set up and run combinator:
else:
    with combinatorInputList1 as combInList1Reg:
        curCombinatorElementPtr1[0].set_to(regPtrVal[combInList1Reg])
    with combinatorInputList2 as combInList2Reg:
        curCombinatorElementPtr2[0].set_to(regPtrVal[combInList2Reg])

    # We initialise the aggregator dependent on the combinator:
    if combinator == 0:  # foldl
        with combinatorStartAcc as combStartAccReg:
            aggregatorIntVal[0].set_to(regIntVal[combStartAccReg])
            aggregatorPtrVal[0].set_to(regPtrVal[combStartAccReg])
            aggregatorBoolVal[0].set_to(regBoolVal[combStartAccReg])
        listIsOver[0].set_to(PtrIsNull(curCombinatorElementPtr1[0],
                                       stackPtrAtCombinatorStart))
    elif combinator == 1:  # map
        aggregatorIntVal[0].set_to(0)
        aggregatorPtrVal[0].set_to(0)
        aggregatorBoolVal[0].set_to(0)
        listIsOver[0].set_to(PtrIsNull(curCombinatorElementPtr1[0],
                                       stackPtrAtCombinatorStart))
    elif combinator == 2:  # zipWith
        aggregatorIntVal[0].set_to(0)
        aggregatorPtrVal[0].set_to(0)
        aggregatorBoolVal[0].set_to(0)
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
                outInt = lambdaRegIntVal[l, i]
                outPtr = lambdaRegPtrVal[l, i]
                outBool = lambdaRegBoolVal[l, i]
                arg1Int = tmpLambdaArg1IntVal[l, i]
                arg1Ptr = tmpLambdaArg1PtrVal[l, i]
                arg1Bool = tmpLambdaArg1BoolVal[l, i]
                arg2Int = tmpLambdaArg2IntVal[l, i]
                arg2Ptr = tmpLambdaArg2PtrVal[l, i]
                arg2Bool = tmpLambdaArg2BoolVal[l, i]
                condition = tmpLambdaConditionVal[l, i]

                # Get the inputs:
                with lambdaInstructionsArg1[i] as r:
                    if r == ele1RegisterIdx:
                        arg1Int.set_to(curCombinatorElementIntVal1[l])
                        arg1Ptr.set_to(0)
                        arg1Bool.set_to(0)
                    elif r == aggRegisterIdx:
                        if combinator == 0:  # foldl
                            arg1Int.set_to(aggregatorIntVal[l])
                            arg1Ptr.set_to(aggregatorPtrVal[l])
                            arg1Bool.set_to(aggregatorBoolVal[l])
                        elif combinator == 1:  # map
                            arg1Int.set_to(0)
                            arg1Ptr.set_to(0)
                            arg1Bool.set_to(0)
                        elif combinator == 2:  # zipWith
                            arg1Int.set_to(curCombinatorElementIntVal2[l])
                            arg1Ptr.set_to(0)
                            arg1Bool.set_to(0)
                    elif r == idxRegisterIdx:
                        arg1Int.set_to(l)
                        arg1Ptr.set_to(0)
                        arg1Bool.set_to(0)
                    elif r < prefixRegisterNum:
                        arg1Int.set_to(regIntVal[r])
                        arg1Ptr.set_to(regPtrVal[r])
                        arg1Bool.set_to(regBoolVal[r])
                    else:
                        arg1Int.set_to(lambdaRegIntVal[l, r - prefixRegisterNum])
                        arg1Ptr.set_to(lambdaRegPtrVal[l, r - prefixRegisterNum])
                        arg1Bool.set_to(lambdaRegBoolVal[l, r - prefixRegisterNum])
                with lambdaInstructionsArg2[i] as r:
                    if r == ele1RegisterIdx:
                        arg2Int.set_to(curCombinatorElementIntVal1[l])
                        arg2Ptr.set_to(0)
                        arg2Bool.set_to(0)
                    elif r == aggRegisterIdx:
                        if combinator == 0:  # foldl
                            arg2Int.set_to(aggregatorIntVal[l])
                            arg2Ptr.set_to(aggregatorPtrVal[l])
                            arg2Bool.set_to(aggregatorBoolVal[l])
                        elif combinator == 1:  # map
                            arg2Int.set_to(0)
                            arg2Ptr.set_to(0)
                            arg2Bool.set_to(0)
                        elif combinator == 2:  # zipWith
                            arg2Int.set_to(curCombinatorElementIntVal2[l])
                            arg2Ptr.set_to(0)
                            arg2Bool.set_to(0)
                    elif r == idxRegisterIdx:
                        arg2Int.set_to(l)
                        arg2Ptr.set_to(0)
                        arg2Bool.set_to(0)
                    elif r < prefixRegisterNum:
                        arg2Int.set_to(regIntVal[r])
                        arg2Ptr.set_to(regPtrVal[r])
                        arg2Bool.set_to(regBoolVal[r])
                    else:
                        arg2Int.set_to(lambdaRegIntVal[l, r - prefixRegisterNum])
                        arg2Ptr.set_to(lambdaRegPtrVal[l, r - prefixRegisterNum])
                        arg2Bool.set_to(lambdaRegBoolVal[l, r - prefixRegisterNum])
                with lambdaInstructionsCondition[i] as r:
                    if r < prefixRegisterNum:
                        condition.set_to(regBoolVal[r])
                    else:
                        condition.set_to(lambdaRegBoolVal[l, r - prefixRegisterNum])

                # Stack pointer:
                # number of iterations we already did * size of the closure body
                # + how far we are in this one:
                curStackPtr = stackPtrAtCombinatorStart + l * (lambdaLength + 1) + i

                ExecuteInstruction(
                    lambdaInstructions[i],
                    arg1Ptr, arg1Int, arg1Bool,
                    arg2Ptr, arg2Int, arg2Bool,
                    condition,
                    outPtr, outInt, outBool,
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
                lambdaIntOut = lambdaRegIntVal[l, outputRegIndex]
                lambdaPtrOut = lambdaRegPtrVal[l, outputRegIndex]
                lambdaBoolOut = lambdaRegBoolVal[l, outputRegIndex]

                if combinator == 0:  # foldl
                    aggregatorIntVal[l + 1].set_to(lambdaIntOut)
                    aggregatorPtrVal[l + 1].set_to(lambdaPtrOut)
                    aggregatorBoolVal[l + 1].set_to(lambdaBoolOut)
                    # These just stay empty:
                    stackCarVal[newStackPtr].set_to(0)
                    stackCdrVal[newStackPtr].set_to(0)
                elif combinator == 1:  # map
                    # We create a new stack element from the result register,
                    # and point to the next element we will create:
                    aggregatorIntVal[l + 1].set_to(0)
                    aggregatorPtrVal[l + 1].set_to(0)
                    aggregatorBoolVal[l + 1].set_to(0)
                    stackCarVal[newStackPtr].set_to(lambdaIntOut)
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
                    aggregatorIntVal[l + 1].set_to(0)
                    aggregatorPtrVal[l + 1].set_to(0)
                    aggregatorBoolVal[l + 1].set_to(0)
                    stackCarVal[newStackPtr].set_to(lambdaIntOut)
                    if listIsOver[l + 1] == 1:
                        stackCdrVal[newStackPtr].set_to(0)
                    elif listIsOver[l + 1] == 0:
                        if l + 1 < maxLoopsteps:
                            stackCdrVal[newStackPtr].set_to(nextGeneratedStackPtr)
                        else:
                            stackCdrVal[newStackPtr].set_to(0)

        elif listIsOver[l] == 1:
            listIsOver[l + 1].set_to(1)
            aggregatorIntVal[l + 1].set_to(aggregatorIntVal[l])
            aggregatorPtrVal[l + 1].set_to(aggregatorPtrVal[l])
            aggregatorBoolVal[l + 1].set_to(aggregatorBoolVal[l])
            curCombinatorElementPtr1[l + 1].set_to(0)
            curCombinatorElementPtr2[l + 1].set_to(0)

            # We still need to initialise the stack cells for all these steps to 0:
            for i in range(0, lambdaLength + 1):
                curStackPtr = stackPtrAtCombinatorStart + l * (lambdaLength + 1) + i
                stackCarVal[curStackPtr].set_to(0)
                stackCdrVal[curStackPtr].set_to(0)

    if combinator == 0:  # foldl:
        regIntVal[prefixRegisterNum].set_to(aggregatorIntVal[maxLoopsteps])
        regPtrVal[prefixRegisterNum].set_to(aggregatorPtrVal[maxLoopsteps])
        regBoolVal[prefixRegisterNum].set_to(aggregatorBoolVal[maxLoopsteps])
    elif combinator == 1:  # map:
        regIntVal[prefixRegisterNum].set_to(0)
        # Point to first list element generated in map:
        regPtrVal[prefixRegisterNum].set_to(stackPtrAtCombinatorStart + lambdaLength)
        regBoolVal[prefixRegisterNum].set_to(0)
    elif combinator == 2:  # zipWith:
        regIntVal[prefixRegisterNum].set_to(0)
        # Point to first list element generated in zipWith:
        regPtrVal[prefixRegisterNum].set_to(stackPtrAtCombinatorStart + lambdaLength)
        regBoolVal[prefixRegisterNum].set_to(0)

##### Run suffix
for i in range(suffixLength):
    # Aliases for instruction processing. Instructions are
    # of the following form: "out = op arg1 arg2", where
    # arg1 and arg2 are either pointers or integers and
    # are chosen based on the type of the operator.
    outInt = regIntVal[prefixRegisterNum + 1 + i]
    outPtr = regPtrVal[prefixRegisterNum + 1 + i]
    outBool = regBoolVal[prefixRegisterNum + 1 + i]
    arg1Int = tmpSuffixArg1IntVal[i]
    arg1Ptr = tmpSuffixArg1PtrVal[i]
    arg1Bool = tmpSuffixArg1BoolVal[i]
    arg2Int = tmpSuffixArg2IntVal[i]
    arg2Ptr = tmpSuffixArg2PtrVal[i]
    arg2Bool = tmpSuffixArg2BoolVal[i]
    condition = tmpSuffixConditionVal[i]

    # Get the inputs:
    with suffixInstructionsArg1[i] as r:
        arg1Int.set_to(regIntVal[r])
        arg1Ptr.set_to(regPtrVal[r])
        arg1Bool.set_to(regBoolVal[r])
    with suffixInstructionsArg2[i] as r:
        arg2Int.set_to(regIntVal[r])
        arg2Ptr.set_to(regPtrVal[r])
        arg2Bool.set_to(regBoolVal[r])
    with suffixInstructionsCondition[i] as r:
        condition.set_to(regBoolVal[r])

    curStackPtr = stackPtrAtSuffixStart + i

    ExecuteInstruction(
        suffixInstructions[i],
        arg1Ptr, arg1Int, arg1Bool,
        arg2Ptr, arg2Int, arg2Bool,
        condition,
        outPtr, outInt, outBool,
        curStackPtr,
        stackCarVal[curStackPtr], stackCdrVal[curStackPtr])


outputTermState.set_to(1)
if expectListOutput == 0:
    # Copy register to output:
    with programReturnReg as outputRegIndex:
        outputRegIntVal.set_to(regIntVal[outputRegIndex])
        outputRegBoolVal.set_to(regBoolVal[outputRegIndex])
    # Set list output bits to default values:
    for n in range(stackSize):
        outputListIsDone[n].set_to(1)
        outputListVal[n].set_to(0)
elif expectListOutput == 1:
    # Set output register value to default:
    outputRegIntVal.set_to(0)
    outputRegBoolVal.set_to(0)
    # Copy list values out:
    outputListCopyPos = Var(stackSize)[stackSize + 1]
    with programReturnReg as outputRegIndex:
        outputListCopyPos[0].set_to(regPtrVal[outputRegIndex])
    for n in range(stackSize):
        outputListIsDone[n].set_to(PtrIsNull(outputListCopyPos[n], stackSize))
        if outputListIsDone[n] == 1:
            outputListVal[n].set_to(0)
            outputListCopyPos[n + 1].set_to(0)

        elif outputListIsDone[n] == 0:
            with outputListCopyPos[n] as p:
                outputListVal[n].set_to(stackCarVal[p])
                outputListCopyPos[n + 1].set_to(stackCdrVal[p])
