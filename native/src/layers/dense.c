#include <include/layer.h>

void layer_dense_init(Layer* layer, unsigned int input_size, unsigned int batches) {
  layer->input_size = input_size;
  layer->weights = matrix_new_randf(input_size, layer->output_size);
  layer->biases = matrix_new(1, layer->output_size, TYPE_F32);
}

void layer_dense_reset(Layer* layer, unsigned int batches) {
  matrix_free(layer->output);
  layer->output = matrix_new(batches, layer->output_size, TYPE_F32);
}

Matrix* layer_dense_feed_forward(Layer* layer, Matrix* input) {
  matrix_dot(input, layer->weights, layer->output);
  float* output_data = (float*) layer->output->data;
  float* bias_data = (float*) layer->biases->data;
  for (int i = 0, j = 0; i < layer->output->rows * layer->output->cols; i++, j++) {
    if (j == layer->output->cols) {
      j = 0;
    }
    output_data[i] = layer->activation->fx(output_data[i] + bias_data[j]);
  }
  return layer->output;
}

void layer_dense_back_prop(Layer* layer, Matrix* input, Matrix* error, float learning_rate) {
  // TODO
}

Layer* layer_dense(unsigned int size, ActivationType activation_type) {
  Layer* layer = malloc(sizeof(Layer));

  layer->input_size = 0; // Set by init
  layer->output_size = size;

  layer->activation = get_activation(activation_type);

  // Set by init
  layer->weights = NULL;
  layer->biases = NULL;
  // Set by reset
  layer->output = NULL;
  
  layer->init = layer_dense_init;
  layer->reset = layer_dense_reset;
  layer->feed_forward = layer_dense_feed_forward;
  layer->back_prop = layer_dense_back_prop;

  layer->data = NULL;
  layer->finalizer = NULL;

  return layer;
}
