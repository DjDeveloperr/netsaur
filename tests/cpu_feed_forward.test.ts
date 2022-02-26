import { NeuralNetwork } from "../mod.ts";
import { CPUMatrix } from "../src/cpu/matrix.ts";
import { CPUNetwork } from "../src/cpu/network.ts";

const net = await new NeuralNetwork({
    hidden: [
        { size: 2, activation: "sigmoid" }
    ],
    cost: "crossentropy",
    output: { size: 2, activation: "sigmoid" }
}).setupBackend(false);

const network = (net.network as CPUNetwork);

network.initialize("f32", 2, 3);

const res = network.feedForward(
    new CPUMatrix(new Float32Array([
        1, 2,
        3, 4,
        5, 6
    ]), 2, 3)
)

console.log(res.data)