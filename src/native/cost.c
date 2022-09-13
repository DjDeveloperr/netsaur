#include "cost.h"
#include "stdlib.h"

CostFx* cost_from_type(CostFunction type) {
  switch (type) {
    // case COST_CROSS_ENTROPY:
    //   return (CostFx*) &CrossEntropy;
    // case COST_HINGE:
    //   return (CostFx*) &Hinge;
    // case COST_MSE:
    //   return (CostFx*) &MeanSquaredError;
    default:
      return NULL;
  }
}
