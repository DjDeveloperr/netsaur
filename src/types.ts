import { DataArray, DataType } from "../deps.ts";

export interface Network<T extends DataType = DataType> {
    addLayers(layer: LayerConfig[]): void
    getOutput(): DataArray<T>
    train(datasets: DataSet<T>, epochs: number, batches: number, learningRate: number): void
}

export interface NetworkConfig {
    input?: InputConfig;
    hidden: LayerConfig[];
    cost: Cost;
    output: LayerConfig;
}

export interface LayerConfig {
    size: number
    activation: Activation
}

export type Activation = "sigmoid" | "tanh" | "relu" | "leakyrelu"

export type Cost = "crossentropy" | "hinge"

export type Shape = number

export type InputConfig = {
    size: number,
    type: DataType
}

export type DataSet<T extends DataType = DataType> = {
    inputs: DataArray<T>
    outputs: DataArray<T>
}