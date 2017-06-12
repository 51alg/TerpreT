module Testcases

open System
open System.IO
open Newtonsoft.Json
open Common
  
type Benchmark = {
    name : string
    hyperparams : IHyperparams;
    seed : int
    examples : Example seq
}

type BenchmarkInstance = {
    batch_name : string
    hypers : string
    instances : obj seq
}

let examplesAreDiverse (exs: Example seq) : bool =
    match (Seq.head exs).output with
        | ListVal _ | IntVal _ -> 
            let outs = Seq.map (fun ex -> ex.output) exs |> Set.ofSeq
            Set.count outs > 1

        // Examples with boolean outputs should have good diversity of true and false
        // results.
        | BoolVal _ ->
            let t_count = Seq.filter (fun ex -> ex.output = BoolVal true) exs |> Seq.length
            let total = Seq.length exs
            let f_count = total - t_count
            t_count > (total / 4) && f_count > (total / 4)

let regenerateUntil (mkEx : Random -> Example) (check : Example -> bool) : (Random -> Example) =
    fun (rand : Random) ->
        let mutable checkPassed = false
        let mutable ex = None
        while not checkPassed do
            if ex.IsSome then
                printfn "  Rejecting example because it didn't pass check."
            ex <- Some (mkEx rand)
            checkPassed <- check ex.Value
        ex.Value

let mkBenchmark (name : string) (hypers : IHyperparams) (seed : int) (exNumber : int) (mkEx : Random -> Example) : Benchmark =
    printfn "Generating %i examples for benchmark %s with seed %i." exNumber name seed
    let rand = Random(seed) in
    
    let examples = Collections.Generic.List<Example>()
    let mutable haveDiverseSet = false
    while not haveDiverseSet do
        examples.Clear()
        for i in 1 .. exNumber do
            let newEx = ref <| mkEx rand
            while Seq.exists (fun oldEx -> oldEx.inputs = (!newEx).inputs) examples do
                printfn "  Rejecting example because inputs were seen before."
                newEx := mkEx rand
            examples.Add(!newEx)
        haveDiverseSet <- examplesAreDiverse examples
        if not haveDiverseSet then
            printfn "  Rejecting example set because all outputs are equal."

    { name = name ;
      hyperparams = hypers ;
      seed = seed ;
      examples = examples }

let writeData (rand : System.Random) (benchs : Benchmark seq) (outDir: string) =
    let hypers_to_data = ref Map.empty
    let benchs = List.ofSeq benchs
    for bench in benchs do
        let hypers = bench.hyperparams

        let convertedBench =
            {
                batch_name = if bench.seed = 0 then "train" else sprintf "train__seed_%i" bench.seed
                hypers = hypers.Name
                instances = Seq.map (hypers.ExampleToRecord rand) bench.examples
            }
        match Map.tryFind hypers !hypers_to_data with
        | Some l ->
            hypers_to_data := Map.add hypers (convertedBench :: l) !hypers_to_data
        | None ->
            hypers_to_data := Map.add hypers [convertedBench] !hypers_to_data

    for (hypers, benchInstances) in Map.toSeq !hypers_to_data do
        let name = (List.head benchs).name
        let baseFilename = name + "__" + hypers.Name + ".data.json"
        let filename = if outDir <> "" then Path.Combine(outDir, baseFilename) else baseFilename
    
        use outStream = new StreamWriter(new FileStream(filename, FileMode.Create))
        fprintfn outStream "%s" (JsonConvert.SerializeObject(benchInstances, Formatting.Indented))

let writeHypers (hypers : IHyperparams) (outDir : string) =
    let baseFilename = hypers.Name + ".hypers.json"
    let filename = if outDir <> "" then Path.Combine(outDir, baseFilename) else baseFilename
    use outStream = new StreamWriter(new FileStream(filename, FileMode.Create))
    fprintfn outStream "%s" (JsonConvert.SerializeObject(Map.ofSeq [(hypers.Name, hypers.Map)], Formatting.Indented))

let inputOutputDiffer (ex : Example) : bool =
    ex.inputs.[0] <> ex.output

let outputListNonEmpty (ex : Example) : bool =
    match ex.output with
    | ListVal [] -> false
    | ListVal _ -> true
    | _ -> raise (ArgumentException(""))

let mkRevBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "rev" hypers seed exNumber
        (regenerateUntil
            (fun rand ->
                let len = rand.Next(1, hypers.InputStackSize + 1)
                let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
                let outList = List.rev inList

                { inputs = [ListVal inList] ;
                  output = ListVal outList })
            (fun ex -> inputOutputDiffer ex && outputListNonEmpty ex))

let mkMapIncBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "mapInc" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(1, hypers.InputStackSize + 1)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let outList = List.map ((+) 1) inList

            { inputs = [ListVal inList] ;
              output = ListVal outList })

let mkRevMapIncBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "revMapInc" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(1, hypers.InputStackSize + 1)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let outList = List.fold (fun acc ele -> (ele + 1)::acc) [] inList

            { inputs = [ListVal inList] ;
              output = ListVal outList })

let mkMapAddKBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "mapAddK" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(1, hypers.InputStackSize + 1)
            let k = rand.Next(0, hypers.MaxInt)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let outList = List.map ((+) k) inList

            { inputs = [ListVal inList ; IntVal k] ;
              output = ListVal outList }) 
 
let mkSumBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "sum" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(1, hypers.InputStackSize + 1)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let res = List.sum inList

            { inputs = [ListVal inList] ;
              output = IntVal res })

let mkAllGtKBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "allGtK" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(1, hypers.InputStackSize + 1)
            let k = rand.Next(0, hypers.MaxInt)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let out = List.forall (fun e -> e > k) inList

            { inputs = [ListVal inList ; IntVal k] ;
              output = BoolVal out })

let mkExGtKBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "exGtK" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(1, hypers.InputStackSize + 1)
            let k = rand.Next(0, hypers.MaxInt)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let out = List.exists (fun e -> e > k) inList

            { inputs = [ListVal inList ; IntVal k] ;
              output = BoolVal out })

let mkFindLastIdxBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "findLastIdx" hypers seed exNumber
        (fun rand ->
            let len1 = rand.Next(1, hypers.InputStackSize - 1)
            let len2 = rand.Next(1, hypers.InputStackSize - len1 - 1)
            let k = rand.Next(0, hypers.MaxInt)
            let inList = 
                (List.init len1 (fun _ -> rand.Next(0, hypers.MaxInt)))
                @ [k]
                @ (List.init len2 (fun _ -> rand.Next(0, hypers.MaxInt)))
            let out = fst <| List.fold (fun (res, idx) ele -> if ele = k then (idx, idx + 1) else (res, idx + 1)) (0, 0) inList

            { inputs = [ListVal inList ; IntVal k] ;
              output = IntVal out })

let mkConcatBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "concat" hypers seed exNumber
        (fun rand ->
            let len1 = rand.Next(1, hypers.InputStackSize)
            let len2 = rand.Next(0, hypers.InputStackSize - len1 + 1)
            let inList1 = List.init len1 (fun _ -> rand.Next(0, hypers.MaxInt))
            let inList2 = List.init len2 (fun _ -> rand.Next(0, hypers.MaxInt))
            let outList = inList1 @ inList2

            { inputs = [ListVal inList1; ListVal inList2] ;
              output = ListVal outList })

let mkMaxBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "max" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(1, hypers.InputStackSize + 1)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let out = List.max inList

            { inputs = [ListVal inList] ;
              output = IntVal out })

let mkLenBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "len" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(1, hypers.InputStackSize + 1)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let out = len

            { inputs = [ListVal inList] ;
              output = IntVal out })

let mkPairwiseSumBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "pairwiseSum" hypers seed exNumber
        (fun rand ->
            let len = hypers.InputStackSize / 2
            let inList1 = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let inList2 = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let outList = List.zip inList1 inList2 |> List.map (fun (x, y) -> x + y)

            { inputs = [ListVal inList1; ListVal inList2] ;
              output = ListVal outList })

let mkGetKBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "getK" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(1, hypers.InputStackSize + 1)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let k = rand.Next(0, len - 1)
            let out = inList.[k]

            { inputs = [ListVal inList; IntVal k] ;
              output = IntVal out } )

let mkLast2Benchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "last2" hypers seed exNumber
        (fun rand ->
            let len = rand.Next(2, hypers.InputStackSize + 1)
            let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
            let out = List.head (List.tail (List.rev inList))

            { inputs = [ListVal inList] ;
              output = IntVal out } )

let mkRevFilterGtKBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "revFilterGtK" hypers seed exNumber
        (regenerateUntil
            (fun rand ->
                let len = rand.Next(1, hypers.InputStackSize + 1)
                let k = rand.Next(0, hypers.MaxInt)
                let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
                let out = List.fold (fun res ele -> if ele > k then ele::res else res) [] inList

                { inputs = [ListVal inList; IntVal k] ;
                  output = ListVal out } )
            (fun ex -> inputOutputDiffer ex && outputListNonEmpty ex))

let mkDup2Benchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
    mkBenchmark "dup2" hypers seed exNumber
        (fun rand ->
            let k = rand.Next(0, hypers.MaxInt)
            let out = k :: k :: [];

            { inputs = [IntVal k] ;
              output = ListVal out } )

let mkRevInterleaveBenchmark (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
  let rec interleave l e = 
    match l with
      | [] | [_] -> l
      | x::xs -> x::e::(interleave xs e)
      
  mkBenchmark "revInterleave" hypers seed exNumber
    (fun rand ->
        let len = rand.Next(1, hypers.InputStackSize + 1)
        let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
        let elem = rand.Next(0, hypers.MaxInt)
        let out = interleave inList elem |> List.rev

        { inputs = [ListVal inList; IntVal elem] ;
          output = ListVal out } )

let mkGetKBenchmarkSL (k : int) (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
  mkBenchmark (sprintf "get%d" k) hypers seed exNumber
    (fun rand ->
        let len = rand.Next(k, hypers.InputStackSize + 1)
        let inList = List.init len (fun _ -> rand.Next(0, hypers.MaxInt))
        let out = inList.[k - 1]

        { inputs = [ListVal inList] ;
          output = IntVal out } )

let mkDupKBenchmarkSL (k : int) (hypers : IHyperparams) (seed : int) (exNumber : int) : Benchmark =
  mkBenchmark (sprintf "dup%d" k) hypers seed exNumber
    (fun rand ->
        let elem = rand.Next(0, hypers.MaxInt)
        let out = List.init k (fun _ -> elem)

        { inputs = [IntVal elem] ;
          output = ListVal out } )
