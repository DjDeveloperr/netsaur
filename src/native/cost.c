#include "cost.h"
#include "stdlib.h"

CostFx* cost_from_type(CostFunction type) {
  switch (type) {
    case COST_CROSS_ENTROPY:
      return (CostFx*) &CrossEntropy;
    // case COST_HINGE:
    //   return (CostFx*) &Hinge;
    // case COST_MSE:
    //   return (CostFx*) &MeanSquaredError;
    default:
      return NULL;
  }
}

float cross_entropy_cost(float* yHat, float* y, unsigned int len) {
  float cost = 0.0;
  for (unsigned int i = 0; i < len; i++) {
    cost += y[i] * log(yHat[i]) + (1.0 - y[i]) * log(1.0 - yHat[i]);
  }
  return -cost;
}

float cross_entropy_prime(float yHat, float y) {
  return yHat - y;
}

const CostFx CrossEntropy = { cross_entropy_cost, cross_entropy_prime };
