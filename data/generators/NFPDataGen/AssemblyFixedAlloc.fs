module AssemblyFixedAlloc

open System
open Common

type Hyperparams(inputNum : int,
                 inputStackSize : int,
                 numRegisters : int,
                 programLength : int,
                 numTimesteps : int) =
  member val MaxScalar = 1 + inputStackSize + numTimesteps
  member val InputNum = inputNum
  member val InputStackSize = inputStackSize
  member val NumRegisters = numRegisters
  member val ProgramLength = programLength
  member val NumTimesteps = numTimesteps
  member self.MaxInt = self.MaxScalar
  member self.StackSize = self.MaxScalar
  
  interface IHyperparams with
    member __.InputNum = inputNum
    member __.InputStackSize = inputStackSize
    member self.StackSize = self.MaxScalar
    member self.MaxInt = self.MaxScalar
    member __.Name =
      [
        sprintf "inputNum_%i" inputNum;
        sprintf "inputStackSize_%i" inputStackSize;
        sprintf "numRegisters_%i" numRegisters;
        sprintf "programLen_%i" programLength;
        sprintf "numTimesteps_%i" numTimesteps;
      ] |> String.concat "__"
        |> fun s -> "asmfxa__" + s
    member __.Map =
      [
        "inputNum", inputNum;
        "inputStackSize", inputStackSize;
        "numRegisters", numRegisters;
        "programLen", programLength;
        "numTimesteps", numTimesteps;
      ] |> Map.ofList

    member self.ExampleToRecord (rand : System.Random) (ex : Example) =
      let (inputRegPtrVals, inputRegIntVals, inputRegBoolVals, stackCarVal, stackCdrVal)
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
      let (outputRegVal, outputListVal) =
        let nullV = System.Nullable ()
        let nullStack = List.replicate self.StackSize nullV
        match ex.output with
        | ListVal l ->
          let len = List.length l
          let l = List.map (fun x -> x % self.MaxInt |> Nullable) l
          let paddedList = l @ (List.replicate (self.StackSize - len) nullV)
          let isDoneList =
              (List.replicate len 0) @ (List.replicate (self.StackSize - len) 1)
              |> List.map Nullable
          (nullV, paddedList)
        | IntVal i -> (i % self.MaxInt |> Nullable, nullStack)
        | BoolVal b -> ((if b then 1 else 0) |> Nullable, nullStack)
      
      {
          inputRegVal  =     regScalarVals
          inputStackCarVal = stackCarVal
          inputStackCdrVal = stackCdrVal

          outputTermState =  1
          outputRegVal =     outputRegVal
          outputListVal =    outputListVal
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
