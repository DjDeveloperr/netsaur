#include "layer.h"
#include "stdlib.h"
#include "stdio.h"

Layer* layer_new(unsigned int outputSize, ActivationFunction activation) {
  Layer* layer = malloc(sizeof(Layer));
  
  layer->outputSize = outputSize;
  layer->activation = activation_from_type(activation);

  layer->input = NULL;
  layer->weights = NULL;
  layer->biases = NULL;
  layer->output = NULL;
  
  return layer;
}

void layer_reset(Layer* layer, unsigned int batches) {
  if (layer->output != NULL) {
    matrix_free(layer->output);
  }
  layer->output = matrix_new(layer->outputSize, batches, TYPE_F32);
}

void layer_init(Layer* layer, unsigned int inputSize, unsigned int batches) {
  layer->weights = matrix_new_randf(layer->outputSize, inputSize);
  layer->biases = matrix_new_fill_f32(layer->outputSize, batches, 0.0);
  layer_reset(layer, batches);
}

Matrix* layer_feed_forward(Layer* layer, Matrix* input) {
  layer->input = input;
  printf("dot %p %p %p\n", input, layer->weights, layer->output);
  matrix_dot(input, layer->weights, layer->output);
  printf("dot done\n");
  printf("add\n");
  matrix_add(layer->output, layer->biases, layer->output);
  printf("add done\n");
  printf("activate\n");
  layer->activation->activate(layer->output);
  printf("activate done\n");
  return layer->output;
}

void layer_free(Layer* layer) {
  // Not owned by Layer
  // if (layer->input != NULL) matrix_free(layer->input);
  if (layer->weights != NULL) matrix_free(layer->weights);
  if (layer->biases != NULL) matrix_free(layer->biases);
  if (layer->output != NULL) matrix_free(layer->output);
  free(layer);
}
