module L2

open System
open System.IO
open Newtonsoft.Json

open Common
open Testcases

type Hyperparams(maxInt : int,
                 inputStackSize : int) =

  interface IHyperparams with
    member x.InputNum = raise (new NotImplementedException ())
    member x.InputStackSize = inputStackSize
    member x.StackSize = raise (new NotImplementedException ())
    member x.MaxInt = maxInt
    member x.Name =
      [
        sprintf "maxInt_%i" maxInt;
        sprintf "inputStackSize_%i" inputStackSize;
      ] |> String.concat "__"
        |> fun s -> "l2__" + s
    member x.Map =
     [
       "maxInt", maxInt;
       "inputStackSize", inputStackSize;
     ] |> Map.ofList

    member self.ExampleToRecord (rand : System.Random) (ex : Example) =
        raise (new NotImplementedException ())

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

type Contents =
  {
    examples : string list;
  }

type BenchmarkInstance =
  {
    name : string;
    kind : string;
    contents : Contents;
    
    batch_name : string;
    hypers : string;
  }

let valueToString =
  function
    | ListVal l -> "[" + (List.map (sprintf "%d") l |> String.concat " ") + "]"
    | IntVal x -> sprintf "%d" x
    | BoolVal true -> "#t"
    | BoolVal false -> "#f"

let exampleToString (name: String) (ex: Example) : string =
  let inputsStr = List.map valueToString ex.inputs |> String.concat " " in
  let outputStr = valueToString ex.output in
  sprintf "(%s %s) -> %s" name inputsStr outputStr

let examplesToString (name: String) (exs: Example seq) : string list =
  Seq.map (exampleToString name) exs
  |> Seq.toList

let writeData (benchs : Benchmark seq) (outDir: string) =
    let hypers_to_data = ref Map.empty
    let benchs = List.ofSeq benchs
    for bench in benchs do
        let hypers = bench.hyperparams

        let convertedBench =
            {
                batch_name = if bench.seed = 0 then "train" else sprintf "train__seed_%i" bench.seed
                hypers = hypers.Name
                name = bench.name
                kind = "examples"
                contents = { examples = examplesToString bench.name bench.examples }
            }
            
        let baseFilename = bench.name + "__" + hypers.Name + ".data.json"
        let filename = if outDir <> "" then Path.Combine(outDir, baseFilename) else baseFilename
    
        use outStream = new StreamWriter(new FileStream(filename, FileMode.Create))
        fprintfn outStream "%s" (JsonConvert.SerializeObject(convertedBench, Formatting.Indented))

