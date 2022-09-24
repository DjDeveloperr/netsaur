#include "layer.h"
#include "stdlib.h"
#include "stdio.h"
#include <math.h>

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
  // Ownership is transfered to JS when returned from a function
  // if (layer->output != NULL) {
  //   matrix_free(layer->output);
  // }
  layer->output = matrix_new(batches, layer->outputSize, TYPE_F32);
}

void layer_init(Layer* layer, unsigned int inputSize, unsigned int batches) {
  layer->weights = matrix_new_randf(inputSize, layer->outputSize);
  layer->biases = matrix_new_fill_f32(batches, layer->outputSize, 0.0);
  layer_reset(layer, batches);
}

Matrix* layer_feed_forward(Layer* layer, Matrix* input) {
  layer->input = input;
  if (matrix_dot(input, layer->weights, layer->output) == NULL) {
    printf("Error: matrix_dot failed %d %d, %d %d, %d %d\n", input->rows, input->cols, layer->weights->rows, layer->weights->cols, layer->output->rows, layer->output->cols);
  }
  float* biasData = layer->biases->data;
  float* outputData = layer->output->data;
  for (int i = 0, j = 0; i < layer->output->rows * layer->output->cols; i++, j++) {
    if (j >= layer->biases->rows) j = 0;
    float sum = outputData[i] + biasData[j];
    outputData[i] = layer->activation->activate(sum);
  }
  return layer->output;
}

void layer_free(Layer* layer) {
  // Not owned by Layer
  // if (layer->input != NULL) matrix_free(layer->input);
  if (layer->weights != NULL) matrix_free(layer->weights);
  if (layer->biases != NULL) matrix_free(layer->biases);
  // Ownership is transfered to JS when returned from a function
  // if (layer->output != NULL) matrix_free(layer->output);
  free(layer);
}
