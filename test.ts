import { Layer, Network } from "./src/native/mod.ts";
import ffi from "./src/native/ffi.ts";

const network = new Network({
  inputSize: 2,
  layers: [
    new Layer({ outputSize: 100, activation: "sigmoid" }),
    new Layer({ outputSize: 1, activation: "sigmoid" }),
  ],
  cost: "crossentropy",
});

network.init(2, 1);

console.log(network.predict(new Float32Array([1, 0])));

Deno.bench(
  "native predict",
  () => {
    ffi.network_predict(network.unsafePointer, new Float32Array([1, 0]), 2);
  },
);
