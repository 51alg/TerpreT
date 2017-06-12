module CombinatorTypedImmutable

open System
open Common

type Hyperparams(maxInt : int,
                 inputNum : int,
                 inputStackSize : int,
                 prefixLength : int,
                 lambdaLength : int,
                 suffixLength : int) =
  member val MaxInt = maxInt
  member val InputNum = inputNum
  member val InputStackSize = inputStackSize
  member val PrefixLength = prefixLength
  member val LambdaLength = lambdaLength
  member val SuffixLength = suffixLength
  member val StackSize =
    if lambdaLength > 0 then 
        1 + inputStackSize + prefixLength + (inputStackSize + prefixLength) * (1 + lambdaLength) + suffixLength
    else
        1 + inputStackSize + prefixLength + suffixLength
  
  interface IHyperparams with
    member x.InputNum = inputNum
    member x.InputStackSize = inputStackSize
    member x.StackSize = x.StackSize
    member x.MaxInt = maxInt
    member x.Name =
      [
        sprintf "maxInt_%i" maxInt;
        sprintf "inputNum_%i" inputNum;
        sprintf "inputStackSize_%i" inputStackSize;
        sprintf "prefixLength_%i" prefixLength;
        sprintf "lambdaLength_%i" lambdaLength;
        sprintf "suffixLength_%i" suffixLength;
      ] |> String.concat "__"
        |> fun s -> "cmbtypimm__" + s
    member x.Map =
     [
       "maxInt", maxInt;
       "inputNum", inputNum;
       "inputStackSize", inputStackSize;
       "prefixLength", prefixLength;
       "lambdaLength", lambdaLength;
       "suffixLength", suffixLength;
     ] |> Map.ofList

    member self.ExampleToRecord (rand : System.Random) (ex : Example) =
      let (inputRegPtrVals, inputRegIntVals, inputRegBoolVals, stackIntVal, stackPtrVal)
        = Common.inputsToRegistersAndStack self rand ex.inputs

      let (outputRegIntVal, outputRegBoolVal, outputListVal, outputListIsDone) =
        match ex.output with
        | ListVal l ->
          let len = List.length l
          let l = List.map (fun x -> x % self.MaxInt) l
          let paddedList = l @ (List.replicate (self.StackSize - len) 0)
          let isDoneList = (List.replicate len 0) @ (List.replicate (self.StackSize - len) 1)
          (0, 0, paddedList, isDoneList)
        | IntVal i ->
          (i % self.MaxInt, 0, List.replicate self.StackSize 0, List.replicate self.StackSize 1)
        | BoolVal b ->
          (0, (if b then 1 else 0), List.replicate self.StackSize 0, List.replicate self.StackSize 1)
      
      {
          inputRegPtrVal =   inputRegPtrVals
          inputRegIntVal =   inputRegIntVals
          inputRegBoolVal =  inputRegBoolVals
          inputStackCarVal = stackIntVal
          inputStackCdrVal = stackPtrVal

          expectListOutput = match ex.output with | ListVal _ -> 1 | _ -> 0
          outputRegIntVal =  outputRegIntVal
          outputRegBoolVal = outputRegBoolVal
          outputListVal =    outputListVal
          outputListIsDone = outputListIsDone
          outputTermState = 1
      } :> _


  override x.Equals (yobj : obj) =
    match yobj with
    | :? Hyperparams as y ->
        (x :> IHyperparams).Map = (y :> IHyperparams).Map 
    | _ -> false

  override x.GetHashCode () = (x :> IHyperparams).Map.GetHashCode()

  interface IComparable with
    member x.CompareTo(y) =
      match y with
        | :? IHyperparams as y -> compare (x :> IHyperparams).Name y.Name
        | _ -> invalidArg "y" "cannot compare values of different types"
