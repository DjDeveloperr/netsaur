#include "activation.h"
#include "stddef.h"
#include "math.h"

ActivationFx* activation_from_type(ActivationFunction type) {
  switch (type) {
    case ACT_SIGMOID:
      return (ActivationFx*) &Sigmoid;
    case ACT_TANH:
      return (ActivationFx*) &Tanh;
    default:
      return NULL;
  }
}

float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float sigmoid_prime(float x) {
  return x * (1.0 - x);
}

const ActivationFx Sigmoid = { sigmoid, sigmoid_prime };

float tanh_prime(float x) {
  return 1.0 - x * x;
}

const ActivationFx Tanh = { tanf, tanh_prime };
