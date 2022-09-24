#pragma once

#include "matrix.h"

typedef struct ActionvationFx {
  float (*activate) (float x);
  float (*prime) (float x);
} ActivationFx;

typedef unsigned char ActivationFunction;
ActivationFx* activation_from_type(ActivationFunction type);

#define ACT_SIGMOID (ActivationFunction) 0
#define ACT_TANH (ActivationFunction) 1
// #define ACT_RELU 2
// #define ACT_LEAKY_RELU 3
// #define ACT_ELU 4
// #define ACT_SELU 5

const ActivationFx Sigmoid;
const ActivationFx Tanh;
// const ActivationFx ReLU;
// const ActivationFx LeakyReLU;
// const ActivationFx Elu;
// const ActivationFx SelU;
