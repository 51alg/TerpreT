module Main

open Common
open Testcases

let benchmarks =
  [ 
    mkRevBenchmark ;
    mkMapIncBenchmark ;
    mkRevMapIncBenchmark ;
    mkMapAddKBenchmark ;
    mkSumBenchmark ;
    mkConcatBenchmark ;
    mkMaxBenchmark ;
    mkLenBenchmark ;
    mkPairwiseSumBenchmark ;
    mkGetKBenchmark ;
    mkLast2Benchmark ;
    mkRevFilterGtKBenchmark ;
    mkFindLastIdxBenchmark ;
    mkDup2Benchmark ;
    mkRevInterleaveBenchmark
  ]

let boolBenchmarks =
  [
    mkAllGtKBenchmark ;
    mkExGtKBenchmark ;
  ]

let straightLineBenchmarks : (IHyperparams -> int -> int -> Benchmark) list ref = ref []

//Declare this first, so that we can define printUsage...
let private options = Mono.Options.OptionSet()

let printUsage outStream =
    fprintfn outStream "Usage: %s" (System.Reflection.Assembly.GetEntryAssembly().GetName().Name)
    options.WriteOptionDescriptions(outStream)

let genHypers maxScalar inputNum inputStackSize prefixLength lambdaLength suffixLength extraRegisterNum numRegisters programLength numTimesteps = 
      [
        CombinatorTypedImmutable.Hyperparams(maxInt = maxScalar,
                                             inputNum = inputNum,
                                             inputStackSize = inputStackSize,
                                             prefixLength = prefixLength,
                                             lambdaLength = lambdaLength,
                                             suffixLength = suffixLength) :> IHyperparams;
        CombinatorTyped.Hyperparams(maxInt = maxScalar,
                                    inputNum = inputNum,
                                    inputStackSize = inputStackSize,
                                    prefixLength = prefixLength,
                                    lambdaLength = lambdaLength,
                                    suffixLength = suffixLength,
                                    extraRegisterNum = extraRegisterNum) :> IHyperparams;
        CombinatorImmutable.Hyperparams(inputNum = inputNum,
                                        inputStackSize = inputStackSize,
                                        prefixLength = prefixLength,
                                        lambdaLength = lambdaLength,
                                        suffixLength = suffixLength) :> IHyperparams;
        Combinator.Hyperparams(inputNum = inputNum,
                               inputStackSize = inputStackSize,
                               prefixLength = prefixLength,
                               lambdaLength = lambdaLength,
                               suffixLength = suffixLength,
                               extraRegisterNum = extraRegisterNum) :> IHyperparams;
        AssemblyLoopsTyped.Hyperparams(maxInt = maxScalar,
                                       inputNum = inputNum,
                                       inputStackSize = inputStackSize,
                                       prefixLength = prefixLength,
                                       loopBodyLength = lambdaLength,
                                       suffixLength = suffixLength,
                                       extraRegisterNum = extraRegisterNum) :> IHyperparams;
        AssemblyLoops.Hyperparams(inputNum = inputNum,
                                  inputStackSize = inputStackSize,
                                  prefixLength = prefixLength,
                                  loopBodyLength = lambdaLength,
                                  suffixLength = suffixLength,
                                  extraRegisterNum = extraRegisterNum) :> IHyperparams;
        Assembly.Hyperparams(maxScalar = maxScalar,
                             inputNum = inputNum,
                             inputStackSize = inputStackSize,
                             numRegisters = numRegisters,
                             programLength = programLength,
                             numTimesteps = numTimesteps) :> IHyperparams;
        AssemblyFixedAlloc.Hyperparams(inputNum = inputNum,
                                       inputStackSize = inputStackSize,
                                       numRegisters = numRegisters,
                                       programLength = programLength,
                                       numTimesteps = numTimesteps) :> IHyperparams;
      ]

let parseOptions arguments =
    let outDir = ref "."
    let exampleNumber = ref 5
    let exampleNumberBool = ref 25
    let seeds = ref [0]
    let maxScalar = ref 10
    let inputNum = ref 2
    let inputStackSize = ref 5
    let prefixLength = ref 0
    let lambdaLength = ref 3
    let suffixLength = ref 2
    let programLength = ref 2
    let numRegisters = ref 5
    let numTimesteps = ref 5
    let extraRegisterNum = ref 1
    let straightLineMaxStack = ref 9
    options
            //The actual run mode:
            .Add( "help"
                , "Show help."
                , fun _ -> printUsage System.Console.Out; System.Environment.Exit 0)
            .Add( "outDir="
                , sprintf "Directory in which to place generated files. [default: %s]" !outDir
                , fun s -> outDir := s)
            .Add( "exampleNumber="
                , sprintf "Number of examples to generate. [default: %i]" !exampleNumber
                , fun i -> exampleNumber := i)
            .Add( "exampleNumberBool="
                , sprintf "Number of examples to generate for boolean programs. [default: %i]" !exampleNumberBool
                , fun i -> exampleNumber := i)
            .Add( "seeds="
                , sprintf "Random seeds for which to generate examples. [default: %s]" (!seeds |> List.map string |> String.concat ", ")
                , fun (s : string) -> seeds := s.Split [| ',' |] |> Seq.map int |> List.ofSeq)
            .Add( "maxScalar="
                , sprintf "The number of scalar values. [default: %i]" !maxScalar
                , fun i -> maxScalar := i)
            .Add( "inputNum="
                , sprintf "The number of inputs. [default: %i]" !inputNum
                , fun i -> inputNum := i)
            .Add( "inputStackSize="
                , sprintf "The size of the input stack. [default: %i]" !inputStackSize
                , fun i -> inputStackSize := i)
            .Add( "prefixLength="
                , sprintf "The number of instructions in prefix (Combinator model). [default: %i]" !prefixLength
                , fun i -> prefixLength := i)
            .Add( "lambdaLength="
                , sprintf "The number of instructions in lambda closure (Combinator model). [default: %i]" !lambdaLength
                , fun i -> lambdaLength := i)
            .Add( "suffixLength="
                , sprintf "The number of instructions in suffix (Combinator model). [default: %i]" !suffixLength
                , fun i -> suffixLength := i)
            .Add( "programLength="
                , sprintf "The number of instructions in program (Assembly model). [default: %i]" !programLength
                , fun i -> programLength := i)
            .Add( "numRegisters="
                , sprintf "The number of registers in program (Assembly model). [default: %i]" !numRegisters
                , fun i -> numRegisters := i)
            .Add( "numTimesteps="
                , sprintf "The number of timesteps of program runs (Assembly model). [default: %i]" !numTimesteps
                , fun i -> numTimesteps := i)
            .Add( "extraRegisterNum="
                , sprintf "The number of registers in program in addition to registers for inputs and lambda arguments (Mutable model). [default: %i]" !extraRegisterNum
                , fun i -> extraRegisterNum := i)
            .Add( "straightLineMaxStack="
                , sprintf "The maximum size of the input stack for a straight line program. [default: %i]" !straightLineMaxStack
                , fun i -> straightLineMaxStack := i)
            .Add( "debugger"
                , "Launch debugger to attach to this process."
                , fun _ -> System.Diagnostics.Debugger.Launch() |> ignore)
            |> ignore
    let remainingArguments = options.Parse arguments
    if not (Seq.isEmpty remainingArguments) then
        eprintfn "Unknown arguments: %s" (String.concat " " remainingArguments)
        printUsage System.Console.Error
        System.Environment.Exit -1

    let hypers = genHypers !maxScalar !inputNum !inputStackSize !prefixLength !lambdaLength !suffixLength !extraRegisterNum !numRegisters !programLength !numTimesteps
    let slHypers = genHypers !maxScalar !inputNum !straightLineMaxStack !prefixLength !lambdaLength !suffixLength !extraRegisterNum !numRegisters !programLength !numTimesteps

    let l2Hypers =
      L2.Hyperparams(maxInt = !maxScalar, inputStackSize = !inputStackSize)

    let l2SLHypers =
      L2.Hyperparams(maxInt = !maxScalar, inputStackSize = !straightLineMaxStack)

    straightLineBenchmarks :=
      [
        [1..!straightLineMaxStack] |> List.map mkGetKBenchmarkSL ;
        [1..!straightLineMaxStack] |> List.map mkDupKBenchmarkSL ;
      ] |> List.concat

    (!outDir, hypers, slHypers, l2Hypers, l2SLHypers, !exampleNumber, !exampleNumberBool, !seeds)

[<EntryPoint>]
let main args =
    let (outDir, hypers, slHypers, l2Hypers, l2SLHypers, exNumber, exNumberBool, seeds) = parseOptions args
    for h in hypers do
      writeHypers h outDir
      for mkBench in benchmarks do
        writeData (System.Random(0)) (seeds |> Seq.map (fun seed -> mkBench h seed exNumber)) outDir
      for mkBench in boolBenchmarks do
        writeData (System.Random(0)) (seeds |> Seq.map (fun seed -> mkBench h seed exNumberBool)) outDir

    for mkBench in benchmarks do
      L2.writeData (seeds |> Seq.map (fun seed -> mkBench l2Hypers seed exNumber)) outDir
    for mkBench in boolBenchmarks do
      L2.writeData (seeds |> Seq.map (fun seed -> mkBench l2Hypers seed exNumberBool)) outDir

    for mkBench in !straightLineBenchmarks do
      L2.writeData (seeds |> Seq.map (fun seed -> mkBench l2SLHypers seed exNumber)) outDir

    for h in slHypers do
      writeHypers h outDir
      for mkBench in !straightLineBenchmarks do
        writeData (System.Random(0)) (seeds |> Seq.map (fun seed -> mkBench h seed exNumber)) outDir
        
    0
