#include <include/network.h>

Network* network_create(
  unsigned int input_size,
  CostType cost_type,
  unsigned int num_layers,
  Layer** layers
) {
  Network* network = malloc(sizeof(Network));

  network->input_size = input_size;
  network->cost = get_cost(cost_type);
  network->num_layers = num_layers;
  network->layers = layers;

  for (unsigned int i = 0; i < network->num_layers; i++) {
    network->layers[i]->init(network->layers[i], input_size);
    input_size = network->layers[i]->output_size;
  }

  return network;
}

void* network_free(Network* network) {
  for (unsigned int i = 0; i < network->num_layers; i++) {
    layer_free(network->layers[i]);
  }
  // TODO: why is this invalid
  // free(network->layers);
  free(network);
}

Matrix* network_feed_forward(Network* network, Matrix* input) {
  for (unsigned int i = 0; i < network->num_layers; i++) {
    network->layers[i]->reset(network->layers[i], input->rows);
  }
  Matrix* output = input;
  for (unsigned int i = 0; i < network->num_layers; i++) {
    output = network->layers[i]->feed_forward(network->layers[i], output);
  }
  return output;
}

void network_back_prop(Network* network, Matrix* target, float learning_rate) {
  Layer* output_layer = network->layers[network->num_layers - 1];
  Matrix* error = matrix_new(output_layer->output->rows, output_layer->output->cols, output_layer->output->type);
  
  float* error_data = (float*) error->data;
  float* output_layer_data = (float*) output_layer->output->data;
  float* target_data = (float*) target->data;
  for (int i = 0; i < output_layer->output->rows * output_layer->output->cols; i++) {
    error_data[i] = network->cost->dfx(output_layer_data[i], target_data[i]);
  }

  output_layer->back_prop(output_layer, error, learning_rate);

  Matrix* weights = output_layer->weights;
  for (int i = network->num_layers - 2; i >= 0; i--) {
    Layer* layer = network->layers[i];
    error = matrix_dot(error, matrix_transpose(weights), NULL);
    layer->back_prop(layer, error, learning_rate);
    weights = layer->weights;
  }
}

void network_train(Network* network, unsigned int num_datasets, Dataset** datasets, unsigned int epochs, float learning_rate) {
  for (unsigned int i = 0; i < epochs; i++) {
    for (unsigned int j = 0; j < num_datasets; j++) {
      Dataset* dataset = datasets[j];
      for (unsigned int i = 0; i < network->num_layers; i++) {
        network->layers[i]->reset(network->layers[i], dataset->inputs->rows);
      }
      network_feed_forward(network, dataset->inputs);
      network_back_prop(network, dataset->outputs, learning_rate);
    }
  }
}
