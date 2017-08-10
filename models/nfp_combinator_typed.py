from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

#### Parameters to the model (changes in this block should not require
#### any changes in the actual model)
maxInt = Hyper()
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

# Ensure that we have enough registers that the learner doesn't have
# to overwrite program inputs or lambda inputs.
registerNum = inputNum + extraRegisterNum
lambdaRegisterNum = registerNum + 3

inputRegIntVal = Input(maxInt)[inputNum]
inputRegPtrVal = Input(stackSize)[inputNum]
inputRegBoolVal = Input(2)[inputNum]
inputStackCarVal = Input(maxInt)[inputStackSize]
inputStackCdrVal = Input(stackSize)[inputStackSize]

#### Outputs:
outputRegIntVal = Output(maxInt)
outputRegBoolVal = Output(2)
outputListVal = Output(maxInt)[stackSize]
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
def And(a, b): return 1 if a == 1 and b == 1 else 0
@Runtime([boolSize, boolSize], boolSize)
def Or(a, b): return 1 if a == 1 or b == 1 else 0

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
stackCarVal = Var(maxInt)[stackSize]
stackCdrVal = Var(stackSize)[stackSize]

## Program registers
regIntVal = Var(maxInt)[numTimesteps, registerNum]
regPtrVal = Var(stackSize)[numTimesteps, registerNum]
regBoolVal = Var(2)[numTimesteps, registerNum]

## Aggregator values, one after each loop iteration (+ an initial value)
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
tmpPrefixOutIntVal = Var(maxInt)[prefixLength]
tmpPrefixOutPtrVal = Var(stackSize)[prefixLength]
tmpPrefixOutBoolVal = Var(2)[prefixLength]
tmpPrefixConditionVal = Var(2)[prefixLength]
tmpPrefixDoWriteReg = Var(2)[prefixLength, registerNum]

tmpLambdaArg1IntVal = Var(maxInt)[maxLoopsteps, lambdaLength]
tmpLambdaArg1PtrVal = Var(stackSize)[maxLoopsteps, lambdaLength]
tmpLambdaArg1BoolVal = Var(2)[maxLoopsteps, lambdaLength]
tmpLambdaArg2IntVal = Var(maxInt)[maxLoopsteps, lambdaLength]
tmpLambdaArg2PtrVal = Var(stackSize)[maxLoopsteps, lambdaLength]
tmpLambdaArg2BoolVal = Var(2)[maxLoopsteps, lambdaLength]
tmpLambdaOutIntVal = Var(maxInt)[maxLoopsteps, lambdaLength]
tmpLambdaOutPtrVal = Var(stackSize)[maxLoopsteps, lambdaLength]
tmpLambdaOutBoolVal = Var(2)[maxLoopsteps, lambdaLength]
tmpLambdaConditionVal = Var(2)[maxLoopsteps, lambdaLength]
tmpLambdaDoWriteReg = Var(2)[maxLoopsteps, lambdaLength, registerNum]

tmpCombinatorOutputDoWriteReg = Var(2)[registerNum]

tmpSuffixArg1IntVal = Var(maxInt)[suffixLength]
tmpSuffixArg1PtrVal = Var(stackSize)[suffixLength]
tmpSuffixArg1BoolVal = Var(2)[suffixLength]
tmpSuffixArg2IntVal = Var(maxInt)[suffixLength]
tmpSuffixArg2PtrVal = Var(stackSize)[suffixLength]
tmpSuffixArg2BoolVal = Var(2)[suffixLength]
tmpSuffixOutIntVal = Var(maxInt)[suffixLength]
tmpSuffixOutPtrVal = Var(stackSize)[suffixLength]
tmpSuffixOutBoolVal = Var(2)[suffixLength]
tmpSuffixConditionVal = Var(2)[suffixLength]
tmpSuffixDoWriteReg = Var(2)[suffixLength, registerNum]

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
    regIntVal[0, i].set_to(inputRegIntVal[i])
    regPtrVal[0, i].set_to(inputRegPtrVal[i])
    regBoolVal[0, i].set_to(inputRegBoolVal[i])
for r in range(inputNum, registerNum):
    regIntVal[0, r].set_to(0)
    regPtrVal[0, r].set_to(0)
    regBoolVal[0, r].set_to(0)

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
    outInt = tmpPrefixOutIntVal[t]
    outPtr = tmpPrefixOutPtrVal[t]
    outBool = tmpPrefixOutBoolVal[t]
    arg1Int = tmpPrefixArg1IntVal[t]
    arg1Ptr = tmpPrefixArg1PtrVal[t]
    arg1Bool = tmpPrefixArg1BoolVal[t]
    arg2Int = tmpPrefixArg2IntVal[t]
    arg2Ptr = tmpPrefixArg2PtrVal[t]
    arg2Bool = tmpPrefixArg2BoolVal[t]
    condition = tmpPrefixConditionVal[t]

    # Get the inputs:
    with prefixInstructionsArg1[t] as r:
        arg1Int.set_to(regIntVal[t, r])
        arg1Ptr.set_to(regPtrVal[t, r])
        arg1Bool.set_to(regBoolVal[t, r])
    with prefixInstructionsArg2[t] as r:
        arg2Int.set_to(regIntVal[t, r])
        arg2Ptr.set_to(regPtrVal[t, r])
        arg2Bool.set_to(regBoolVal[t, r])
    with prefixInstructionsCondition[t] as r:
        condition.set_to(regBoolVal[t, r])

    curStackPtr = stackPtrAtPrefixStart + t

    ExecuteInstruction(
        prefixInstructions[t],
        arg1Ptr, arg1Int, arg1Bool,
        arg2Ptr, arg2Int, arg2Bool,
        condition,
        outPtr, outInt, outBool,
        curStackPtr,
        stackCarVal[curStackPtr], stackCdrVal[curStackPtr])

    for r in range(registerNum):
        tmpPrefixDoWriteReg[t, r].set_to(RegEqTest(prefixInstructionsOut[t], r))
        if tmpPrefixDoWriteReg[t, r] == 0:
            regIntVal[t + 1, r].set_to(regIntVal[t, r])
            regPtrVal[t + 1, r].set_to(regPtrVal[t, r])
            regBoolVal[t + 1, r].set_to(regBoolVal[t, r])
        elif tmpPrefixDoWriteReg[t, r] == 1:
            regIntVal[t + 1, r].set_to(outInt)
            regPtrVal[t + 1, r].set_to(outPtr)
            regBoolVal[t + 1, r].set_to(outBool)


t = prefixLength
##### If we have no body, do nothing:
if lambdaLength == 0:
    for r in range(registerNum):
        tmpCombinatorOutputDoWriteReg[r].set_to(RegEqTest(combinatorOut, r))
        if tmpCombinatorOutputDoWriteReg[r] == 0:
            regPtrVal[t + 1, r].set_to(regPtrVal[t, r])
            regBoolVal[t + 1, r].set_to(regBoolVal[t, r])
            regIntVal[t + 1, r].set_to(regIntVal[t, r])
        elif tmpCombinatorOutputDoWriteReg[r] == 1:
            regIntVal[t + 1, r].set_to(0)
            regPtrVal[t + 1, r].set_to(0)
            regBoolVal[t + 1, r].set_to(0)

##### Otherwise, set up and run combinator:
else:
    ele1RegisterIdx = registerNum
    aggRegisterIdx = registerNum + 1
    idxRegisterIdx = registerNum + 2

    with combinatorInputList1 as combInList1Reg:
        curCombinatorElementPtr1[0].set_to(regPtrVal[t, combInList1Reg])
    with combinatorInputList2 as combInList2Reg:
        curCombinatorElementPtr2[0].set_to(regPtrVal[t, combInList2Reg])

    # We initialise the aggregator dependent on the combinator:
    if combinator == 0:  # foldl
        with combinatorStartAcc as combStartAccReg:
            aggregatorIntVal[0].set_to(regIntVal[t, combStartAccReg])
            aggregatorPtrVal[0].set_to(regPtrVal[t, combStartAccReg])
            aggregatorBoolVal[0].set_to(regBoolVal[t, combStartAccReg])
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
                regPtrVal[t + 1, r].set_to(regPtrVal[t, r])
                regBoolVal[t + 1, r].set_to(regBoolVal[t, r])
                regIntVal[t + 1, r].set_to(regIntVal[t, r])
            for r in range(inputNum, registerNum):
                regPtrVal[t + 1, r].set_to(0)
                regBoolVal[t + 1, r].set_to(0)
                regIntVal[t + 1, r].set_to(0)

            # Execute the body of our lambda:
            for i in range(0, lambdaLength):
                t = prefixLength + ((lambdaLength + 1) * l) + 1 + i

                # Aliases for instruction processing. Instructions are
                # of the following form: "out = op arg1 arg2", where
                # arg1 and arg2 are either pointers or integers and
                # are chosen based on the type of the operator.
                outInt = tmpLambdaOutIntVal[l, i]
                outPtr = tmpLambdaOutPtrVal[l, i]
                outBool = tmpLambdaOutBoolVal[l, i]
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
                    else:
                        arg1Int.set_to(regIntVal[t, r])
                        arg1Ptr.set_to(regPtrVal[t, r])
                        arg1Bool.set_to(regBoolVal[t, r])
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
                    else:
                        arg2Int.set_to(regIntVal[t, r])
                        arg2Ptr.set_to(regPtrVal[t, r])
                        arg2Bool.set_to(regBoolVal[t, r])
                with lambdaInstructionsCondition[i] as r:
                    condition.set_to(regBoolVal[t, r])

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

                for r in range(registerNum):
                    tmpLambdaDoWriteReg[l, i, r].set_to(RegEqTest(lambdaInstructionsOut[i], r))
                    if tmpLambdaDoWriteReg[l, i, r] == 0:
                        regIntVal[t + 1, r].set_to(regIntVal[t, r])
                        regPtrVal[t + 1, r].set_to(regPtrVal[t, r])
                        regBoolVal[t + 1, r].set_to(regBoolVal[t, r])
                    elif tmpLambdaDoWriteReg[l, i, r] == 1:
                        regIntVal[t + 1, r].set_to(outInt)
                        regPtrVal[t + 1, r].set_to(outPtr)
                        regBoolVal[t + 1, r].set_to(outBool)

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
                lambdaIntOut = regIntVal[t, outputRegIndex]
                lambdaPtrOut = regPtrVal[t, outputRegIndex]
                lambdaBoolOut = regBoolVal[t, outputRegIndex]

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
                # Copy register forwards.
                t = prefixLength + ((lambdaLength + 1) * l) + i
                for r in range(registerNum):
                    regPtrVal[t + 1, r].set_to(regPtrVal[t, r])
                    regBoolVal[t + 1, r].set_to(regBoolVal[t, r])
                    regIntVal[t + 1, r].set_to(regIntVal[t, r])

                curStackPtr = stackPtrAtCombinatorStart + l * (lambdaLength + 1) + i
                stackCarVal[curStackPtr].set_to(0)
                stackCdrVal[curStackPtr].set_to(0)

    t = prefixLength + ((lambdaLength + 1) * maxLoopsteps)
    for r in range(registerNum):
        tmpCombinatorOutputDoWriteReg[r].set_to(RegEqTest(combinatorOut, r))
        if tmpCombinatorOutputDoWriteReg[r] == 0:
            regPtrVal[t + 1, r].set_to(regPtrVal[prefixLength, r])
            regBoolVal[t + 1, r].set_to(regBoolVal[prefixLength, r])
            regIntVal[t + 1, r].set_to(regIntVal[prefixLength, r])
        elif tmpCombinatorOutputDoWriteReg[r] == 1:
            if combinator == 0:  # foldl
                regIntVal[t+1, r].set_to(aggregatorIntVal[maxLoopsteps])
                regPtrVal[t+1, r].set_to(aggregatorPtrVal[maxLoopsteps])
                regBoolVal[t+1, r].set_to(aggregatorBoolVal[maxLoopsteps])
            elif combinator == 1:  # map:
                regIntVal[t+1, r].set_to(0)
                # Point to first list element generated in map:
                regPtrVal[t+1, r].set_to(stackPtrAtCombinatorStart + lambdaLength)
                regBoolVal[t+1, r].set_to(0)
            elif combinator == 2:  # zipWith:
                regIntVal[t+1, r].set_to(0)
                # Point to first list element generated in zipWith:
                regPtrVal[t+1, r].set_to(stackPtrAtCombinatorStart + lambdaLength)
                regBoolVal[t+1, r].set_to(0)

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
    outInt = tmpSuffixOutIntVal[i]
    outPtr = tmpSuffixOutPtrVal[i]
    outBool = tmpSuffixOutBoolVal[i]
    arg1Int = tmpSuffixArg1IntVal[i]
    arg1Ptr = tmpSuffixArg1PtrVal[i]
    arg1Bool = tmpSuffixArg1BoolVal[i]
    arg2Int = tmpSuffixArg2IntVal[i]
    arg2Ptr = tmpSuffixArg2PtrVal[i]
    arg2Bool = tmpSuffixArg2BoolVal[i]
    condition = tmpSuffixConditionVal[i]

    # Get the inputs:
    with suffixInstructionsArg1[i] as r:
        arg1Int.set_to(regIntVal[t, r])
        arg1Ptr.set_to(regPtrVal[t, r])
        arg1Bool.set_to(regBoolVal[t, r])
    with suffixInstructionsArg2[i] as r:
        arg2Int.set_to(regIntVal[t, r])
        arg2Ptr.set_to(regPtrVal[t, r])
        arg2Bool.set_to(regBoolVal[t, r])
    with suffixInstructionsCondition[i] as r:
        condition.set_to(regBoolVal[t, r])

    curStackPtr = stackPtrAtSuffixStart + i

    ExecuteInstruction(
        suffixInstructions[i],
        arg1Ptr, arg1Int, arg1Bool,
        arg2Ptr, arg2Int, arg2Bool,
        condition,
        outPtr, outInt, outBool,
        curStackPtr,
        stackCarVal[curStackPtr], stackCdrVal[curStackPtr])

    for r in range(registerNum):
        tmpSuffixDoWriteReg[i, r].set_to(RegEqTest(suffixInstructionsOut[i], r))
        if tmpSuffixDoWriteReg[i, r] == 0:
            regIntVal[t + 1, r].set_to(regIntVal[t, r])
            regPtrVal[t + 1, r].set_to(regPtrVal[t, r])
            regBoolVal[t + 1, r].set_to(regBoolVal[t, r])
        elif tmpSuffixDoWriteReg[i, r] == 1:
            regIntVal[t + 1, r].set_to(outInt)
            regPtrVal[t + 1, r].set_to(outPtr)
            regBoolVal[t + 1, r].set_to(outBool)


outputTermState.set_to(1)
# Copy register to output:
with programReturnReg as outputRegIndex:
    outputRegIntVal.set_to(regIntVal[numTimesteps - 1, outputRegIndex])
    outputRegBoolVal.set_to(regBoolVal[numTimesteps - 1, outputRegIndex])

# Copy list values out:
outputListCopyPos = Var(stackSize)[stackSize + 1]
with programReturnReg as outputRegIndex:
    outputListCopyPos[0].set_to(regPtrVal[numTimesteps - 1, outputRegIndex])
for n in range(stackSize):
    with outputListCopyPos[n] as p:
        outputListVal[n].set_to(stackCarVal[p])
        outputListCopyPos[n + 1].set_to(stackCdrVal[p])
