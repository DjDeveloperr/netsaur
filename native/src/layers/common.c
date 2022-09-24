#include <include/layer.h>

void layer_free(Layer* layer) {
  if (layer->finalizer != NULL) {
    layer->finalizer(layer->data);
  }
  matrix_free(layer->weights);
  matrix_free(layer->biases);
  matrix_free(layer->output);
  free(layer);
}
