import { Layer, Matrix, Network } from "./src/native/mod.ts";

const network = new Network({
  inputSize: 2,
  layers: [
    new Layer({ outputSize: 2, activation: "sigmoid" }),
    new Layer({ outputSize: 2, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

network.init(2, 1);

console.log(network);

console.log(network.predict(new Matrix(2, 2, new Float32Array([1, 1]))));
