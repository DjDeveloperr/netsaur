import { CPUNetwork } from "./src/cpu/network.ts";

const net = new CPUNetwork({
  input: {
    type: "f32",
    size: 2,
  },
  hidden: [
    {
      size: 2,
      activation: "sigmoid",
    },
    {
      size: 1,
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

Deno.bench("noop", () => {});
Deno.bench(
  "cpu predict",
  (): any => net.predict(new Float32Array([1, 0, 0, 0, 1, 1, 0, 1])),
);
