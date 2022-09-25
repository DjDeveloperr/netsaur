import { Layer, Network } from "../src/native/mod.ts";
import { loadDataset } from "./common.ts";

const network = new Network({
  inputSize: 784,
  layers: [
    Layer.dense({ units: 16, activation: "sigmoid" }),
    Layer.dense({ units: 10, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

console.log("Loading training dataset...");
const trainSet = loadDataset("train-images.idx", "train-labels.idx");

console.log("Training...");
network.train(trainSet, 20, 0.1);
console.log("Training complete!");

network.save("digit_model.bin");
