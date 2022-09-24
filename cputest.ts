import { CPUNetwork } from "./src/cpu/network.ts";

const net = new CPUNetwork({
  input: {
    type: "f32",
    size: 2,
  },
  hidden: [
    {
      size: 100,
      activation: "sigmoid",
    },
  ],
  output: {
    size: 1,
    activation: "sigmoid",
  },
  cost: "crossentropy",
});

net.initialize("f32", 2, 1);

console.log(net.predict(new Float32Array([1, 0])));

Deno.bench(
  "cpu predict",
  () => {
    net.predict(new Float32Array([1, 0]));
  },
);
