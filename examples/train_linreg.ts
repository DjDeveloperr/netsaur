import { NeuralNetwork } from "../mod.ts";

const time = Date.now();

const net = await new NeuralNetwork({
  silent: true,
  hidden: [],
  cost: "crossentropy",
  output: { size: 1, activation: "relu" },
  input: {
    size: 1,
    type: "f32",
  },
}).setupBackend(true);

await net.train(
  [
    { inputs: [1, 2, 3, 4], outputs: [1, 3, 5, 7] },
  ],
  5000,
  4,
  0.1,
);

console.log(await net.predict(new Float32Array([5])));
console.log(Date.now() - time);
