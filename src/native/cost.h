#pragma once

#include "matrix.h"
#include <math.h>

typedef struct CostFx {
  float (*cost) (float* yHat, float* y, unsigned int len);
  float (*prime) (float yHat, float y);
} CostFx;

typedef unsigned char CostFunction;
CostFx* cost_from_type(CostFunction type);

#define COST_CROSS_ENTROPY (CostFunction) 0
#define COST_HINGE (CostFunction) 1
#define COST_MSE (CostFunction) 2

const CostFx CrossEntropy;
const CostFx Hinge;
const CostFx MeanSquaredError;
