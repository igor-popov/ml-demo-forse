
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
open System.IO
open XPlot.GoogleCharts
open FSharp.Data

// we create a type based on sample data
type Dataset = CsvProvider<"day.csv">
type Datapoint = Dataset.Row

// we can now read data...
let train = Dataset.Load("data/train.csv")
let test = Dataset.Load("data/test.csv")

type Hypothesis = Datapoint -> float

open MathNet
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.LinearAlgebra.DenseVector
open MathNet.Numerics.Random

type Vec = Vector<float>
type Mat = Matrix<float>

let predict (theta:Vec) (v:Vec) = theta * v;

type Featurizer = Datapoint -> float list

let featurizerMain (obs:Datapoint) = 
    let workingday = System.Convert.ToInt32(obs.Workingday);
    [   1.0; 
        float obs.Weekday;
        float workingday;
        float obs.Atemp;
        float obs.Hum;
        float obs.Windspeed];

let featurizerOne (obs:Datapoint) = 
    let workingday = System.Convert.ToInt32(obs.Workingday);
    [   1.0; 
        float obs.Weekday;
        ];

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

let runForAllFeatures (iterations: int) =
    let optimal = solve featurizerMain train.Rows 0.1 iterations;
    let (averageDiff, maxDiff) = validate optimal featurizerMain test.Rows;
    printfn "Average deviation: %f" averageDiff;
    printfn "Max deviation: %f" maxDiff;

    optimal;


let iterationsCount = 100000;
let thetaOptimal = runForAllFeatures(iterationsCount);

let trainCost = calculateCost thetaOptimal featurizerMain train.Rows;
let testCost = calculateCost thetaOptimal featurizerMain test.Rows;

printfn "Using iterations: %d" iterationsCount;
printfn "Train cost: %f" trainCost;
printfn "Test cost: %f" testCost;

let featureIndex = 4;
let dataAndPrediction = test.Rows |> Seq.map (fun day -> {Features = featurizerMain(day); Data = day; Prediction = (vector(featurizerMain(day)) * thetaOptimal);}) |> Seq.toList;
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

    
