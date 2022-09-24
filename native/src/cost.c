#include <include/cost.h>

float cost_cross_entropy(float* y, float* y_hat, unsigned int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += y[i] * log(y_hat[i]);
  }
  return -sum;
}

float cost_cross_entropy_dfx(float y, float y_hat) {
  return y_hat - y;
}

const Cost CrossEntropy = {
  .fx = cost_cross_entropy,
  .dfx = cost_cross_entropy_dfx
};

const Cost* get_cost(CostType type) {
  switch (type) {
    case COST_CROSS_ENTROPY:
      return &CrossEntropy;
    default:
      return NULL;
  }
}
