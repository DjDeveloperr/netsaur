#include <include/layer.h>

void layer_free(Layer* layer) {
  if (layer->finalizer != NULL) {
    layer->finalizer(layer->data);
  }
  if (layer->weights != NULL) matrix_free(layer->weights);
  if (layer->biases != NULL) matrix_free(layer->biases);
  if (layer->output != NULL) matrix_free(layer->output);
  free(layer);
}
