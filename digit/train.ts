import { Layer, Network } from "../src/native/mod.ts";
import { loadDataset } from "./common.ts";

const network = new Network({
  inputSize: 784,
  layers: [
    Layer.dense({ units: 28 * 2, activation: "sigmoid" }),
    Layer.dense({ units: 10, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

console.log("Loading training dataset...");
const trainSet = loadDataset("train-images.idx", "train-labels.idx");

const epochs = 50;
console.log("Training (" + epochs + " epochs)...");
network.train(trainSet, epochs, 0.1);
console.log("Training complete!");

network.save("digit_model.bin");
