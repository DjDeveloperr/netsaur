#pragma once

#include "layer.h"
#include "stdlib.h"
#include "cost.h"

typedef struct {
  unsigned int inputSize;
  unsigned int numLayers;
  Layer** layers;
  CostFx* cost;
} Network;

Network* network_new(unsigned int inputSize, Layer** layers, unsigned int numLayers, CostFunction cost);
void network_free(Network* network);

void network_init(Network* network, unsigned int inputSize, unsigned int batches);

Matrix* network_feed_forward(Network* network, Matrix* input);
Matrix* network_back_propagate(Network* network, Matrix* yHat, float rate);

void network_train(Network* network, Matrix* xs, Matrix* ys, unsigned int epochs, unsigned int batchSize, float rate);
Matrix* network_predict(Network* network, Matrix* xs);
