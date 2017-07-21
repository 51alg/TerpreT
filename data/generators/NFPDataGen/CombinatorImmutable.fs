module CombinatorImmutable

open System
open Common

type Hyperparams(inputNum : int,
                 inputStackSize : int,
                 prefixLength : int,
                 lambdaLength : int,
                 suffixLength : int) =
  member val InputNum = inputNum
  member val InputStackSize = inputStackSize
  member val PrefixLength = prefixLength
  member val LambdaLength = lambdaLength
  member val SuffixLength = suffixLength
  member val MaxScalar =
    if lambdaLength > 0 then 
        1 + inputStackSize + prefixLength + (inputStackSize + prefixLength) * (1 + lambdaLength) + suffixLength
    else
        1 + inputStackSize + prefixLength + suffixLength
  
  interface IHyperparams with
    member x.InputNum = inputNum
    member x.InputStackSize = inputStackSize
    member x.StackSize = x.MaxScalar
    member x.MaxInt = x.MaxScalar
    member x.Name =
      [
        sprintf "inputNum_%i" inputNum;
        sprintf "inputStackSize_%i" inputStackSize;
        sprintf "prefixLength_%i" prefixLength;
        sprintf "lambdaLength_%i" lambdaLength;
        sprintf "suffixLength_%i" suffixLength;
      ] |> String.concat "__"
        |> fun s -> "cmbimm__" + s
    member x.Map =
     [
       "inputNum", inputNum;
       "inputStackSize", inputStackSize;
       "prefixLength", prefixLength;
       "lambdaLength", lambdaLength;
       "suffixLength", suffixLength;
     ] |> Map.ofList

    member self.ExampleToRecord (rand : System.Random) (ex : Example) =
      let (inputRegPtrVals, inputRegIntVals, inputRegBoolVals, stackIntVal, stackPtrVal)
        = Common.inputsToRegistersAndStack self rand ex.inputs

      let regScalarVals =
        [ for (regPtrVal, regIntVal, regBoolVal) in Seq.zip3 inputRegPtrVals inputRegIntVals inputRegBoolVals do
            if regPtrVal <> 0 then
              if regIntVal <> 0 && regBoolVal <> 0 then
                failwith "Several scalar types set for register."
              yield regPtrVal
            elif regIntVal <> 0 then
              if regBoolVal <> 0 then
                failwith "Several scalar types set for register."
              yield regIntVal
            elif regBoolVal <> 0 then
              yield regBoolVal
            else
              yield 0]
      let (outputRegVal, outputListVal, outputListIsDone) =
        let nullV = System.Nullable ()
        let nullStack = List.replicate self.MaxScalar nullV
        match ex.output with
        | ListVal l ->
          let len = List.length l
          let l = List.map (fun x -> x % self.MaxScalar) l
          let paddedList =
              l @ (List.replicate (self.MaxScalar - len) 0)
              |> List.map Nullable
          let isDoneList =
              (List.replicate len 0) @ (List.replicate (self.MaxScalar - len) 1)
              |> List.map Nullable
          (nullV, paddedList, isDoneList)
        | IntVal i ->
          (i % self.MaxScalar |> Nullable, nullStack, nullStack)
        | BoolVal b ->
          ((if b then 1 else 0) |> Nullable, nullStack, nullStack)

      {
          inputRegVal =      regScalarVals
          inputStackCarVal = stackIntVal
          inputStackCdrVal = stackPtrVal

          expectListOutput = match ex.output with | ListVal _ -> 1 | _ -> 0
          outputRegVal =     outputRegVal
          outputListVal =    outputListVal
          outputListIsDone = outputListIsDone
          outputTermState =  1
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
