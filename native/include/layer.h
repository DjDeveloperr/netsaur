#pragma once

#include <include/matrix.h>
#include <include/activation.h>

typedef struct Layer {
  unsigned int input_size;
  unsigned int output_size;

  const Activation* activation;

  Matrix* weights;
  Matrix* biases;
  Matrix* output;

  void (*init) (struct Layer* layer, unsigned int input_size, unsigned int batches);
  void (*reset) (struct Layer* layer, unsigned int batches);
  Matrix* (*feed_forward) (struct Layer* layer, Matrix* input);
  void (*back_prop) (struct Layer* layer, Matrix* input, Matrix* error, float learning_rate);
  
  // Generic layer data
  void* data;
  void (*finalizer) (void* data);
} Layer;

void layer_free(Layer* layer);

Layer* layer_dense(unsigned int size, ActivationType activation_type);
