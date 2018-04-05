
#I "packages/XPlot.GoogleCharts/lib/net45"
#I "packages/Newtonsoft.Json/lib/net40"
#I "packages/FSharp.Data/lib/net40"
#I "packages/Google.DataTable.Net.Wrapper/lib"
#I "packages/MathNet.Numerics/lib/net40"
#I "packages/MathNet.Numerics.FSharp/lib/net40"

#r "FSharp.Data.dll"
#r "XPlot.GoogleCharts.dll"
#r "Google.DataTable.Net.Wrapper.dll"
#r "Newtonsoft.Json.dll"
#r "MathNet.Numerics.dll"
#r "MathNet.Numerics.FSharp.dll"

open System
open XPlot.GoogleCharts
open FSharp.Data

// we create a type based on sample data
type Dataset = CsvProvider<"day.csv">
type Datapoint = Dataset.Row

// we can now read data...
let train = Dataset.Load("data/train.csv")
let test = Dataset.Load("data/test.csv")

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Random

type Vec = Vector<float>
type Mat = Matrix<float>


type Featurizer = Datapoint -> float list



let seed (n: int) = 
    let randomValues = Random.doubles n |> Seq.map (fun d -> 100.0 * d) |> Seq.toList;
    vector randomValues;
    
let gradientStep (theta0: Vec) (X: Mat) (Y: Vec) (alpha: float) = 
    let predictions =  X * theta0;
    let diffVector = predictions - Y;
    
    let theta = theta0 - alpha / (float Y.Count) * (diffVector * X);
    theta;

let rec gradientDescend (theta0: Vec) (X: Mat) (Y: Vec) (alpha: float) (step: int) = 
    let theta = gradientStep theta0 X Y alpha;
    if step <= 1 then theta
    else gradientDescend theta X Y alpha (step-1);

let readData (f:Featurizer) (data:Datapoint seq) =
    let Yt, Xt = 
        data
        |> Seq.toList
        |> List.map (fun obs -> float obs.Cnt, f obs)
        |> List.unzip;
    let X = matrix Xt;
    let Y = vector Yt;
    (X, Y);

let solve (f:Featurizer) (data:Datapoint seq) (alpha: float) (iterations: int) =
    let (X, Y) = readData f data; 
    let theta0 = seed(X.ColumnCount);
    
    let theta = gradientDescend theta0 X Y alpha iterations;
    theta;

let calculateCost (theta:Vec) (f:Featurizer) (data:Datapoint seq) = 
    let (X, Y) = readData f data; 
    let diff = X * theta - Y;
    let costValue = (diff * diff)/(2.0 * (float diff.Count));
    costValue;
  

let validate (theta:Vec) (f:Featurizer) (data:Datapoint seq) = 
    let (X, Y) = readData f data; 
    let Predictions = X * theta;
    let firstPrediction = Predictions.Item(0);
    let firstObservation = Y.Item(0);
    printfn "First prediction: %f" firstPrediction;
    printfn "First observation: %f" firstObservation;
    let Diff =  Predictions - Y;
    let maxDiff = Diff.AbsoluteMaximum();
    let averageDiff = Diff |> Seq.map (fun d -> Math.Abs(d)) |> Seq.average;
    (averageDiff, maxDiff)

type DataAndPrediction = {
    Features: float list;
    Data: Datapoint; 
    Prediction: float;
};

type CalculationResult = {theta: Vec; trainCost: float; testCost: float; iterations: int}

let runForAllFeatures (iterations: int) (f:Featurizer): CalculationResult =
    let optimal = solve f train.Rows 0.1 iterations;
    let (averageDiff, maxDiff) = validate optimal f test.Rows;
    
    let trainCost = calculateCost optimal f train.Rows;
    let testCost = calculateCost optimal f test.Rows;

    {theta = optimal; trainCost = trainCost; testCost=testCost; iterations=iterations};

let featurizerMain (obs:Datapoint) = 
    [   1.0; 
        float obs.Weekday;
        (if obs.Holiday then 1.0 else 0.0) ;
        float obs.Weathersit;
        float obs.Atemp;
        float obs.Atemp * float obs.Atemp;
        float obs.Windspeed * float obs.Windspeed;
        float obs.Hum * float obs.Hum;
        float obs.Hum * float obs.Windspeed;
        float obs.Hum;
        float (Math.Sqrt (float obs.Temp));
        float obs.Windspeed;
    ];

let featureIndex = 10;
let iterationsCount = 60000;
let result = runForAllFeatures iterationsCount featurizerMain;

printfn "Using iterations: %d" iterationsCount;
printfn "Train cost: %f" result.trainCost;
printfn "Test cost: %f" result.testCost;
printfn "Theta values";
for v in result.theta do
    printfn "Theta: %f" v;


let dataAndPrediction = test.Rows |> Seq.map (fun day -> {Features = featurizerMain(day); Data = day; Prediction = (vector(featurizerMain(day)) * result.theta);}) |> Seq.toList;
let orderedDataAndPrediction = dataAndPrediction |> Seq.sortBy (fun day -> day.Features.Item(featureIndex)) |> Seq.toList;
  
dataAndPrediction
    |> Seq.map (fun day -> day.Features.Item(featureIndex), day.Data.Cnt)
    |> Chart.Scatter
    |> Chart.Show;
   
Chart.Line [
    [ for obs in orderedDataAndPrediction -> obs.Features.Item(featureIndex), float obs.Data.Cnt ]
    [ for obs in dataAndPrediction -> obs.Features.Item(featureIndex), obs.Prediction ]
    ]
|> Chart.Show;

let solutionsByIteration =  
    [10..50]
    |> Seq.map (fun i -> (runForAllFeatures i featurizerMain))
    |> Seq.toList

Chart.Line [
    [ for s in solutionsByIteration -> float s.iterations, s.trainCost ]
    [ for s in solutionsByIteration -> float s.iterations, s.testCost ]
    ]
|> Chart.Show;
