import ffi from "./ffi.ts";
import { Layer } from "./layer.ts";
import { Matrix } from "./matrix.ts";

const {
  network_free,
  network_new,
  network_predict,
  network_init,
} = ffi;

const NetworkFinalizer = new FinalizationRegistry(
  (network: Deno.PointerValue) => {
    // network_free(network);
  },
);

enum C_COST {
  crossentropy = 0,
  mse = 1,
}

export type Cost = keyof typeof C_COST;

export interface NetworkConfig {
  inputSize: number;
  layers: Layer[];
  cost: Cost;
}

export class Network {
  #ptr: Deno.PointerValue;
  #token: { ptr: Deno.PointerValue } = { ptr: 0 };

  get unsafePointer() {
    return this.#ptr;
  }

  constructor(config: NetworkConfig) {
    this.#ptr = network_new(
      config.inputSize,
      new BigUint64Array(config.layers.map((e) => BigInt(e.unsafePointer))),
      config.layers.length,
      C_COST[config.cost],
    );
    this.#token.ptr = this.#ptr;
    NetworkFinalizer.register(this, this.#ptr, this.#token);
  }

  init(inputSize: number, batchSize: number): void {
    network_init(this.#ptr, inputSize, batchSize);
  }

  predict(input: Matrix<"f32">): Matrix<"f32"> {
    return new Matrix(network_predict(this.#ptr, input.unsafePointer));
  }

  free(): void {
    if (this.#ptr) {
      network_free(this.#ptr);
      this.#ptr = 0;
      NetworkFinalizer.unregister(this.#token);
    }
  }
}
