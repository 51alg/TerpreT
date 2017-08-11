from dummy import Hyper, Param, Var, Runtime, Input, Output, Inline

#### Parameters to the model (changes in this block should not require
#### any changes in the actual model)
maxInt = Hyper()
inputNum = Hyper()
inputStackSize = Hyper()
prefixLength = Hyper()
loopBodyLength = Hyper()
suffixLength = Hyper()
extraRegisterNum = Hyper()

#### Inputs:
##We first need to work out the size of the stack:
#The loop can run at most for the number of input elements + what was allocated in the prefix:
maxLoopsteps = inputStackSize + prefixLength
#One initial timestep, prefix, the loop, suffix:
numTimesteps = 1 + prefixLength + (loopBodyLength * maxLoopsteps) + suffixLength

# The number of stack cells is dependent on the number of instructions and inputs as follows:
#  - 1 for Nil
#  - inputStackSize
#  - prefixLength (as each instruction can allocate a cell)
#  - maxLoopsteps (as we create one cell after each iteration)
#  - maxLoopsteps * loopBodyLength (as each loopBody instruction can allocate a cell)
#  - suffixLength (as each instruction can allocate a cell)
stackPtrAtPrefixStart = 1 + inputStackSize
stackPtrAtLoopStart = stackPtrAtPrefixStart + prefixLength
stackSize = stackPtrAtLoopStart + maxLoopsteps * loopBodyLength + suffixLength

# Registers allow the inputs, the extras, and three additional ones in the loop:
registerNum = inputNum + extraRegisterNum
loopRegisterNum = registerNum + 2

inputRegIntVal = Input(maxInt)[inputNum]
inputRegPtrVal = Input(stackSize)[inputNum]
inputRegBoolVal = Input(2)[inputNum]
inputStackIntVal = Input(maxInt)[inputStackSize]
inputStackPtrVal = Input(stackSize)[inputStackSize]

#### Outputs
outputRegIntVal = Output(maxInt)
outputRegPtrVal = Var(stackSize) #Data structure output is special, see end of file
outputRegBoolVal = Output(2)
outputListVal = Output(maxInt)[stackSize]

#### Execution model description
## Loops: foreach in l, foreach in zip l1, l2
numLoops = 2

#### Instructions: cons, cdr, car, nil/zero/false, add, inc, eq, gt, and, ite, one/true, noop / copy, dec, or
numInstructions = 14
boolSize = 2
@Runtime([maxInt, maxInt], maxInt)
def Add(x, y): return (x + y) % maxInt
@Runtime([maxInt], maxInt)
def Inc(x): return (x + 1) % maxInt
@Runtime([maxInt], maxInt)
def Dec(x): return (x - 1) % maxInt
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
def PtrIsNull(ptr, curStackPtr): return 1 if (ptr == 0 or ptr >= curStackPtr) else 0
@Runtime([stackSize, stackSize, stackSize+1], boolSize)
def OnePtrIsNull(ptr1, ptr2, curStackPtr): return 1 if (ptr1 == 0 or ptr1 >= curStackPtr) or (ptr2 == 0 or ptr2 >= curStackPtr) else 0

## Prefix instructions and arguments
prefixInstructions = Param(numInstructions)[prefixLength]
prefixInstructionsArg1 = Param(registerNum)[prefixLength]
prefixInstructionsArg2 = Param(registerNum)[prefixLength]
prefixInstructionsCondition = Param(registerNum)[prefixLength]
prefixInstructionsOut = Param(registerNum)[prefixLength]

## Suffix instructions and arguments.
suffixInstructions = Param(numInstructions)[suffixLength]
suffixInstructionsArg1 = Param(registerNum)[suffixLength]
suffixInstructionsArg2 = Param(registerNum)[suffixLength]
suffixInstructionsCondition = Param(registerNum)[suffixLength]
suffixInstructionsOut = Param(registerNum)[suffixLength]

## Choosing the loop, its instructions and their arguments:
loop = Param(numLoops)
loopInputList1 = Param(registerNum)
loopInputList2 = Param(registerNum)

loopBodyInstructions = Param(numInstructions)[loopBodyLength]
loopBodyInstructionsOut = Param(registerNum)[loopBodyLength]
loopBodyInstructionsArg1 = Param(loopRegisterNum)[loopBodyLength]
loopBodyInstructionsArg2 = Param(loopRegisterNum)[loopBodyLength]
loopBodyInstructionsCondition = Param(registerNum)[loopBodyLength]

#### Execution data description
## Stack
stackIntVal = Var(maxInt)[stackSize]
stackPtrVal = Var(stackSize)[stackSize]

## Program registers
regIntVal = Var(maxInt)[numTimesteps, registerNum]
regPtrVal = Var(stackSize)[numTimesteps, registerNum]
regBoolVal = Var(2)[numTimesteps, registerNum]

## Pointers to the current loop element, and values:
curLoopElementPtr1 = Var(stackSize)[maxLoopsteps + 1]
curLoopElementPtr2 = Var(stackSize)[maxLoopsteps + 1]
curLoopElementVal1 = Var(maxInt)[maxLoopsteps]
curLoopElementVal2 = Var(maxInt)[maxLoopsteps]

## Temporary things:
# Temp variable that marks that we've reached the end of the list (and
# just sit out the remaining loop steps)
listIsOver = Var(boolSize)[maxLoopsteps + 1]

# Temp variables containing the input arguments (to simplify the remainder of the model)
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

tmpLoopBodyArg1IntVal = Var(maxInt)[maxLoopsteps, loopBodyLength]
tmpLoopBodyArg1PtrVal = Var(stackSize)[maxLoopsteps, loopBodyLength]
tmpLoopBodyArg1BoolVal = Var(2)[maxLoopsteps, loopBodyLength]
tmpLoopBodyArg2IntVal = Var(maxInt)[maxLoopsteps, loopBodyLength]
tmpLoopBodyArg2PtrVal = Var(stackSize)[maxLoopsteps, loopBodyLength]
tmpLoopBodyArg2BoolVal = Var(2)[maxLoopsteps, loopBodyLength]
tmpLoopBodyOutIntVal = Var(maxInt)[maxLoopsteps, loopBodyLength]
tmpLoopBodyOutPtrVal = Var(stackSize)[maxLoopsteps, loopBodyLength]
tmpLoopBodyOutBoolVal = Var(2)[maxLoopsteps, loopBodyLength]
tmpLoopBodyConditionVal = Var(2)[maxLoopsteps, loopBodyLength]
tmpLoopBodyDoWriteReg = Var(2)[maxLoopsteps, loopBodyLength, registerNum]

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
                       curStackPtr, outPtrStack, outIntStack):
    #Do the actual execution. Every instruction sets its
    #corresponding register value, and the two heap cells:
    if instruction == 0:  # cons
        outInt.set_to(0)
        outPtr.set_to(curStackPtr)
        outBool.set_to(0)
        outIntStack.set_to(arg1Int)
        outPtrStack.set_to(arg2Ptr)
    elif instruction == 1:  # car
        with arg1Ptr as p:
            if p < curStackPtr:
                outInt.set_to(stackIntVal[p])
            else:
                outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(0)
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 2:  # cdr
        outInt.set_to(0)
        with arg1Ptr as p:
            if p < curStackPtr:
                outPtr.set_to(stackPtrVal[p])
            else:
                outPtr.set_to(0)
        outBool.set_to(0)
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 3:  # nil/zero/false
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(0)
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 4:  # add
        outInt.set_to(Add(arg1Int, arg2Int))
        outPtr.set_to(0)
        outBool.set_to(0)
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 5:  # inc
        outInt.set_to(Inc(arg1Int))
        outPtr.set_to(0)
        outBool.set_to(0)
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 6:  # eq
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(EqTest(arg1Int, arg2Int))
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 7:  # gt
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(GtTest(arg1Int, arg2Int))
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 8:  # and
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(And(arg1Bool, arg2Bool))
        #These just stay empty:
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
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 10:  # one/true
        outInt.set_to(1)
        outPtr.set_to(0)
        outBool.set_to(1)
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 11:  # noop/copy
        outInt.set_to(arg1Int)
        outPtr.set_to(arg1Ptr)
        outBool.set_to(arg1Bool)
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 12:  # dec
        outInt.set_to(Dec(arg1Int))
        outPtr.set_to(0)
        outBool.set_to(0)
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)
    elif instruction == 13:  # or
        outInt.set_to(0)
        outPtr.set_to(0)
        outBool.set_to(Or(arg1Bool, arg2Bool))
        #These just stay empty:
        outIntStack.set_to(0)
        outPtrStack.set_to(0)

##### Setting up inputs:
#Copy input registers to temporary registers, set extras to 0:
for i in range(inputNum):
    regIntVal[0, i].set_to(inputRegIntVal[i])
    regPtrVal[0, i].set_to(inputRegPtrVal[i])
    regBoolVal[0, i].set_to(inputRegBoolVal[i])
for r in range(inputNum, registerNum):
    regIntVal[0, r].set_to(0)
    regPtrVal[0, r].set_to(0)
    regBoolVal[0, r].set_to(0)

#Initialize nil element at bottom of stack:
stackIntVal[0].set_to(0)
stackPtrVal[0].set_to(0)

#Copy input stack into our temporary representation:
for i in range(inputStackSize):
    stackIntVal[1 + i].set_to(inputStackIntVal[i])
    stackPtrVal[1 + i].set_to(inputStackPtrVal[i])

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

    #Get the inputs:
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
        stackPtrVal[curStackPtr], stackIntVal[curStackPtr])

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
##### Set up and run loop:
with loopInputList1 as loopList1Reg:
    curLoopElementPtr1[0].set_to(regPtrVal[t, loopList1Reg])
with loopInputList2 as loopList2Reg:
    curLoopElementPtr2[0].set_to(regPtrVal[t, loopList2Reg])

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
    # At each iteration, we run the loopBody, but first extract current list elements:
    if listIsOver[l] == 0:
        with curLoopElementPtr1[l] as curPtr1:
            if curPtr1 < stackPtrAtLoopStart + l * loopBodyLength:
                curLoopElementVal1[l].set_to(stackIntVal[curPtr1])
            else:
                curLoopElementVal1[l].set_to(0)
        with curLoopElementPtr2[l] as curPtr2:
            if curPtr2 < stackPtrAtLoopStart + l * loopBodyLength:
                curLoopElementVal2[l].set_to(stackIntVal[curPtr2])
            else:
                curLoopElementVal2[l].set_to(0)

        #Execute the body of our loopBody:
        for i in range(0, loopBodyLength):
            t = prefixLength + l * loopBodyLength + i
            # Aliases for instruction processing. Instructions are
            # of the following form: "out = op arg1 arg2", where
            # arg1 and arg2 are either pointers or integers and
            # are chosen based on the type of the operator.
            outInt = tmpLoopBodyOutIntVal[l, i]
            outPtr = tmpLoopBodyOutPtrVal[l, i]
            outBool = tmpLoopBodyOutBoolVal[l, i]
            arg1Int = tmpLoopBodyArg1IntVal[l, i]
            arg1Ptr = tmpLoopBodyArg1PtrVal[l, i]
            arg1Bool = tmpLoopBodyArg1BoolVal[l, i]
            arg2Int = tmpLoopBodyArg2IntVal[l, i]
            arg2Ptr = tmpLoopBodyArg2PtrVal[l, i]
            arg2Bool = tmpLoopBodyArg2BoolVal[l, i]
            condition = tmpLoopBodyConditionVal[l, i]

            #Get the inputs:
            with loopBodyInstructionsArg1[i] as r:
                if r == ele1RegisterIdx:
                    arg1Int.set_to(curLoopElementVal1[l])
                    arg1Ptr.set_to(0)
                    arg1Bool.set_to(0)
                elif r == ele2RegisterIdx:
                    arg1Int.set_to(curLoopElementVal2[l])
                    arg1Ptr.set_to(0)
                    arg1Bool.set_to(0)
                else:
                    arg1Int.set_to(regIntVal[t, r])
                    arg1Ptr.set_to(regPtrVal[t, r])
                    arg1Bool.set_to(regBoolVal[t, r])
            with loopBodyInstructionsArg2[i] as r:
                if r == ele1RegisterIdx:
                    arg2Int.set_to(curLoopElementVal1[l])
                    arg2Ptr.set_to(0)
                    arg2Bool.set_to(0)
                elif r == ele2RegisterIdx:
                    arg2Int.set_to(curLoopElementVal2[l])
                    arg2Ptr.set_to(0)
                    arg2Bool.set_to(0)
                else:
                    arg2Int.set_to(regIntVal[t, r])
                    arg2Ptr.set_to(regPtrVal[t, r])
                    arg2Bool.set_to(regBoolVal[t, r])
            with loopBodyInstructionsCondition[i] as r:
                condition.set_to(regBoolVal[t, r])

            #Stack pointer: number of full iterations we already
            #did * size of the loopBody body + how far we are in
            #this one:
            curStackPtr = stackPtrAtLoopStart + l * loopBodyLength + i

            ExecuteInstruction(
                loopBodyInstructions[i],
                arg1Ptr, arg1Int, arg1Bool,
                arg2Ptr, arg2Int, arg2Bool,
                condition,
                outPtr, outInt, outBool,
                curStackPtr,
                stackPtrVal[curStackPtr], stackIntVal[curStackPtr])

            for r in range(registerNum):
                tmpLoopBodyDoWriteReg[l, i, r].set_to(RegEqTest(loopBodyInstructionsOut[i], r))
                if tmpLoopBodyDoWriteReg[l, i, r] == 0:
                    regIntVal[t+1, r].set_to(regIntVal[t, r])
                    regPtrVal[t+1, r].set_to(regPtrVal[t, r])
                    regBoolVal[t+1, r].set_to(regBoolVal[t, r])
                elif tmpLoopBodyDoWriteReg[l, i, r] == 1:
                    regIntVal[t+1, r].set_to(outInt)
                    regPtrVal[t+1, r].set_to(outPtr)
                    regBoolVal[t+1, r].set_to(outBool)

        #Move list pointer for next round already:
        stackPtrAtLoopBodyEnd = stackPtrAtLoopStart + (l + 1) * loopBodyLength - 1
        with curLoopElementPtr1[l] as curElePtr1:
            if curElePtr1 < stackPtrAtLoopBodyEnd:
                curLoopElementPtr1[l + 1].set_to(stackPtrVal[curElePtr1])
            else:
                curLoopElementPtr1[l + 1].set_to(0)
        with curLoopElementPtr2[l] as curElePtr2:
            if curElePtr2 < stackPtrAtLoopBodyEnd:
                curLoopElementPtr2[l + 1].set_to(stackPtrVal[curElePtr2])
            else:
                curLoopElementPtr2[l + 1].set_to(0)

        #Check if the next list element is empty:
        if loop == 0:  # foreach
            listIsOver[l + 1].set_to(PtrIsNull(curLoopElementPtr1[l + 1],
                                               stackPtrAtLoopBodyEnd))
        elif loop == 1:  # foreach zip
            listIsOver[l + 1].set_to(OnePtrIsNull(curLoopElementPtr1[l + 1],
                                                  curLoopElementPtr2[l + 1],
                                                  stackPtrAtLoopBodyEnd))

    elif listIsOver[l] == 1:
        listIsOver[l + 1].set_to(1)
        curLoopElementPtr1[l + 1].set_to(0)
        curLoopElementPtr2[l + 1].set_to(0)

        #We still need to initialise the stack cells for all these steps to 0:
        for i in range(0, loopBodyLength):
            # Copy register forwards.
            t = prefixLength + l * loopBodyLength + i
            for r in range(registerNum):
                regPtrVal[t + 1, r].set_to(regPtrVal[t, r])
                regBoolVal[t + 1, r].set_to(regBoolVal[t, r])
                regIntVal[t + 1, r].set_to(regIntVal[t, r])

            curStackPtr = stackPtrAtLoopStart + l * loopBodyLength + i
            stackIntVal[curStackPtr].set_to(0)
            stackPtrVal[curStackPtr].set_to(0)

##### Run suffix
stackPtrAtSuffixStart = stackPtrAtLoopStart + maxLoopsteps * loopBodyLength
for i in range(suffixLength):
    t = prefixLength + loopBodyLength * maxLoopsteps + i

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

    #Get the inputs:
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
        stackPtrVal[curStackPtr], stackIntVal[curStackPtr])

    for r in range(registerNum):
        tmpSuffixDoWriteReg[i, r].set_to(RegEqTest(suffixInstructionsOut[i], r))
        if tmpSuffixDoWriteReg[i, r] == 0:
            regIntVal[t+1, r].set_to(regIntVal[t, r])
            regPtrVal[t+1, r].set_to(regPtrVal[t, r])
            regBoolVal[t+1, r].set_to(regBoolVal[t, r])
        elif tmpSuffixDoWriteReg[i, r] == 1:
            regIntVal[t+1, r].set_to(outInt)
            regPtrVal[t+1, r].set_to(outPtr)
            regBoolVal[t+1, r].set_to(outBool)


#Copy registers to output:
outputRegIntVal.set_to(regIntVal[numTimesteps - 1, registerNum - 1])
outputRegPtrVal.set_to(regPtrVal[numTimesteps - 1, registerNum - 1])
outputRegBoolVal.set_to(regBoolVal[numTimesteps - 1, registerNum - 1])

#Copt stack to output
outputListCopyPos = Var(stackSize)[stackSize + 1]
outputListCopyPos[0].set_to(outputRegPtrVal)

for n in range(stackSize):
    with outputListCopyPos[n] as p:
        outputListVal[n].set_to(stackIntVal[p])
        outputListCopyPos[n + 1].set_to(stackPtrVal[p])
