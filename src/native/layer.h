#pragma once

#include "activation.h"

typedef struct {
  unsigned int outputSize;
  Matrix* input;
  Matrix* weights;
  Matrix* biases;
  Matrix* output;
  ActivationFx* activation;  
} Layer;

Layer* layer_new(unsigned int outputSize, ActivationFunction activation);
void layer_free(Layer* layer);

void layer_reset(Layer* layer, unsigned int batches);
void layer_init(Layer* layer, unsigned int inputSize, unsigned int batches);

Matrix* layer_feed_forward(Layer* layer, Matrix* input);
void layer_back_propagate(Layer* layer, Matrix* error, float rate);
