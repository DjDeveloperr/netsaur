import { Layer, Matrix, Network } from "./src/native/mod.ts";

const start = Date.now();

const network = new Network({
  inputSize: 2,
  layers: [
    new Layer({ outputSize: 3, activation: "sigmoid" }),
    new Layer({ outputSize: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

network.train(
  [
    {
      inputs: new Matrix(
        4,
        2,
        new Float32Array([
          0,
          0,
          0,
          1,
          1,
          0,
          1,
          1,
        ]),
      ),
      outputs: new Matrix(
        4,
        1,
        new Float32Array([
          0,
          1,
          1,
          0,
        ]),
      ),
    },
  ],
  5000,
  0.1,
);

console.log(network.predict(
  new Matrix(
    4,
    2,
    new Float32Array([
      1,
      0,
      0,
      1,
      0,
      0,
      1,
      1,
    ]),
  ),
));

console.log(Date.now() - start);
