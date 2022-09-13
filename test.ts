import { Layer, Matrix, Network } from "./src/native/mod.ts";

const network = new Network({
  inputSize: 2,
  layers: [
    new Layer({ outputSize: 2, activation: "sigmoid" }),
    new Layer({ outputSize: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

network.init(2, 1);

console.log(network);

Deno.bench("noop", () => {});
Deno.bench(
  "native predict",
  (): any =>
    network.predict(
      new Matrix<"f32">(4, 2, new Float32Array([1, 0, 0, 0, 1, 1, 0, 1, 0, 0])),
    ),
);
