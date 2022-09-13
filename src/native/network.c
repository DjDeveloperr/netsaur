#include "network.h"
#include "stdio.h"
#include "stdlib.h"

Network* network_new(unsigned int inputSize, Layer** layers, unsigned int numLayers, CostFunction cost) {
  Network* network = malloc(sizeof(Network));
  network->inputSize = inputSize;
  network->numLayers = numLayers;
  network->layers = malloc(sizeof(Layer*) * numLayers);
  for (int i = 0; i < numLayers; i++) {
    network->layers[i] = layers[i];
  }
  network->cost = cost_from_type(cost);
  return network;
}

void network_free(Network* network) {
  for (unsigned int i = 0; i < network->numLayers; i++) {
    layer_free(network->layers[i]);
  }
  free(network->layers);
  free(network);
}

void network_init(Network* network, unsigned int inputSize, unsigned int batches) {
  for (unsigned int i = 0; i < network->numLayers; i++) {
    Layer* layer = network->layers[i];
    layer_init(layer, inputSize, batches);
    inputSize = layer->outputSize;
  }
}

Matrix* network_feed_forward(Network* network, Matrix* input) {
  Matrix* output = input;
  for (unsigned int i = 0; i < network->numLayers; i++) {
    output = layer_feed_forward(network->layers[i], output);
  }
  return output;
}

Matrix* network_predict(Network* network, Matrix* xs) {
  for (int i = 0; i < network->numLayers; i++) {
    layer_reset(network->layers[i], 1);
  }
  return network_feed_forward(network, xs);
}
