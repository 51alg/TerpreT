module Common

open System

type Value =
  | ListVal of int list
  | IntVal of int
  | BoolVal of bool

type Example = {
    inputs : Value list
    output : Value
}

type NFPExampleInstance = {
    inputRegVal : int seq
    inputStackCarVal : int seq
    inputStackCdrVal : int seq

    expectListOutput : int
    outputRegVal : Nullable<int>
    outputListVal : Nullable<int> seq
    outputTermState : int
}

type TypedNFPExampleInstance = {
    inputRegPtrVal : int seq
    inputRegIntVal : int seq
    inputRegBoolVal : int seq
    inputStackCarVal : int seq
    inputStackCdrVal : int seq

    expectListOutput : int
    outputRegIntVal : Nullable<int>
    outputRegBoolVal : Nullable<int>
    outputListVal : Nullable<int> seq
    outputTermState : int    
}

type IHyperparams =
  inherit IComparable
  
  abstract member InputNum : int
  abstract member InputStackSize : int
  abstract member StackSize : int
  abstract member MaxInt : int
  abstract member Name : string
  abstract member Map : Map<string, int>

  abstract member ExampleToRecord : System.Random -> Example -> obj

let private shuffle (rand : System.Random) (lst : 'a seq) =
    let lst = Array.ofSeq lst
    let swap i j =
        let item = lst.[i]
        lst.[i] <- lst.[j]
        lst.[j] <- item
    let len = lst.Length
    [0..(len - 2)] |> Seq.iter (fun i -> swap i (rand.Next(i, len)))
    lst

let inputsToRegistersAndStack (hypers : IHyperparams) (rand : System.Random) (inputs : Value list) =
    let inputs =
        let inputNum = List.length inputs
        if inputNum < hypers.InputNum then
            inputs @ List.replicate (hypers.InputNum - inputNum) (IntVal 0)
        else if inputNum = hypers.InputNum then
            inputs
        else
            failwithf "Hypers only allow %i inputs, but example has %i inputs." hypers.InputNum inputNum

    let stackIntVal = Array.create hypers.InputStackSize 0
    let stackPtrVal = Array.create hypers.InputStackSize 0
    let stackPtr = ref 0
    let (inputRegPtrVals, inputRegIntVals, inputRegBoolVals) =
        [
            for input in inputs do
                match input with
                | ListVal l ->
                    let lLength = List.length l
                    let lWithAddrs =
                        Seq.zip l (shuffle rand [!stackPtr .. !stackPtr + lLength - 1])
                        |> Array.ofSeq
                    for i in 0 .. (lLength - 1) do
                        let (ele, addr) = lWithAddrs.[i]
                        let nextAddr =
                           if i + 1 < lLength then
                               snd lWithAddrs.[i + 1] + 1 //Account for nil inserted by model
                           else
                               0
                        stackIntVal.[addr] <- ele % hypers.MaxInt
                        stackPtrVal.[addr] <- nextAddr
                    let startPtr = if lLength = 0 then 0 else (snd lWithAddrs.[0]) + 1 //Acount for nil inserted by model
                    stackPtr := !stackPtr + lLength
                    yield (startPtr, 0, 0)
                | IntVal i ->
                    yield (0, i % hypers.MaxInt, 0)
                | BoolVal b ->
                    yield (0, 0, if b then 1 else 0)
        ] |> List.unzip3

    (inputRegPtrVals, inputRegIntVals, inputRegBoolVals, stackIntVal, stackPtrVal)
