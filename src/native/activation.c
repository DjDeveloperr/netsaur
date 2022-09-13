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

float sigmoid_f32(float x) {
  return 1.0 / (1.0 + exp(-x));
}

Matrix* sigmoid(Matrix* x) {
  if (x->type != TYPE_F32) return NULL;
  matrix_map_f32(x, sigmoid_f32);
  return x;
}

float sigmoid_prime_f32(float x) {
  return x * (1.0 - x);
}

Matrix* sigmoid_prime(Matrix* x) {
  if (x->type != TYPE_F32) return NULL;
  Matrix* result = matrix_copy(x);
  matrix_map_f32(result, sigmoid_prime_f32);
  return result;
}

const ActivationFx Sigmoid = { sigmoid, sigmoid_prime };

Matrix* tanh_activate(Matrix* x) {
  if (x->type != TYPE_F32) return NULL;
  Matrix* result = matrix_copy(x);
  matrix_map_f32(result, tanf);
  return result;
}

float tanh_prime_f32(float x) {
  return 1.0 - x * x;
}

Matrix* tanh_prime(Matrix* x) {
  if (x->type != TYPE_F32) return NULL;
  Matrix* result = matrix_copy(x);
  matrix_map_f32(result, tanh_prime_f32);
  return result;
}

const ActivationFx Tanh = { tanh_activate, tanh_prime };
