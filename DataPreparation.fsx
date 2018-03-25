(*
This file is there for reference, but not important.
This script was used to generate two samples, by
shuffling the original data set (day.csv) and 
splitting it into training and test sets.
*)

let seed = 123456
let rng = System.Random(seed)

// Fischer-Yates shuffle
let shuffle (arr:'a []) =
    let arr = Array.copy arr
    let l = arr.Length
    for i in (l-1) .. -1 .. 1 do
        let temp = arr.[i]
        let j = rng.Next(0,i+1)
        arr.[i] <- arr.[j]
        arr.[j] <- temp
    arr

open System
open System.IO

let rootPath = __SOURCE_DIRECTORY__

let data = 
    rootPath + @"/day.csv"
    |> File.ReadAllLines

let createSamples () =
    let header = data.[0]
    let content = data.[1..] |> shuffle
    let train = Array.append [| header |] content.[..499]
    let test =  Array.append [| header |] content.[500..]
    File.WriteAllLines(rootPath + "/train.csv", train)
    File.WriteAllLines(rootPath + "/test.csv", test)
    