#pragma once

#include <include/layer.h>

typedef struct Network {
  unsigned int input_size;
  unsigned int num_layers;
  Layer** layers;
} Network;

Network* network_create(unsigned int input_size, unsigned int num_layers, Layer** layers);

void* network_free(Network* network);

void network_init(Network* network);
