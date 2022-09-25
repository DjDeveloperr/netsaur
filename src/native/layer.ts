import ffi from "./ffi.ts";

const {
  layer_free,
  layer_dense,
} = ffi;

enum C_ACTIVATION {
  sigmoid = 0,
  tanh = 1,
}

export type Activation = keyof typeof C_ACTIVATION;

const LayerFinalizer = new FinalizationRegistry((layer: Deno.PointerValue) => {
  // layer_free(layer);
});

export interface DenseLayerConfig {
  activation: Activation;
  units: number;
}

export class Layer {
  #ptr: Deno.PointerValue;
  #token: { ptr: Deno.PointerValue } = { ptr: 0 };

  get unsafePointer() {
    return this.#ptr;
  }

  constructor(ptr: Deno.PointerValue) {
    this.#ptr = ptr;
    this.#token.ptr = ptr;
    LayerFinalizer.register(this, this.#ptr, this.#token);
  }

  static dense(config: DenseLayerConfig): Layer {
    return new Layer(
      layer_dense(config.units, C_ACTIVATION[config.activation]),
    );
  }

  free(): void {
    if (this.#ptr) {
      layer_free(this.#ptr);
      this.#ptr = 0;
      LayerFinalizer.unregister(this.#token);
    }
  }
}
