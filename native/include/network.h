#pragma once

#include <include/layer.h>
#include <include/cost.h>

typedef struct Network {
  unsigned int input_size;
  unsigned int num_layers;
  Layer** layers;
  const Cost* cost;
} Network;

Network* network_create(
  unsigned int input_size,
  CostType cost_type,
  unsigned int num_layers,
  Layer** layers
);

void* network_free(Network* network);

void network_init(Network* network);

Matrix* network_feed_forward(Network* network, Matrix* input);

void network_back_prop(Network* network, Matrix* target, float learning_rate);

typedef struct Dataset {
  unsigned int batches;
  Matrix* inputs;
  Matrix* outputs;
} Dataset;

void network_train(Network* network, unsigned int num_datasets, Dataset** datasets, unsigned int epochs, float learning_rate);
