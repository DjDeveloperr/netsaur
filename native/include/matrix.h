#pragma once

#include <stdint.h>
#include <include/util.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define TYPE_U32 0
#define TYPE_I32 1
#define TYPE_F32 2

typedef struct {
  void* data;
  unsigned int rows;
  unsigned int cols;
  unsigned char type;
} Matrix;

Matrix* matrix_new(int rows, int cols, char type);
Matrix* matrix_new_randf(int rows, int cols);

Matrix* matrix_new_from_array(int rows, int cols, char type, void* data);
Matrix* matrix_new_from_array_zero_copy(int rows, int cols, char type, void* data);

Matrix* matrix_new_fill_u32(int rows, int cols, uint32_t v);
Matrix* matrix_new_fill_i32(int rows, int cols, int32_t v);
Matrix* matrix_new_fill_f32(int rows, int cols, float v);

Matrix* matrix_copy(Matrix* m);

Matrix* matrix_map_f32(Matrix* m, float (*f) (float));
Matrix* matrix_map_i32(Matrix* m, int32_t (*f) (int32_t));
Matrix* matrix_map_u32(Matrix* m, uint32_t (*f) (uint32_t));

Matrix* matrix_dot(Matrix* a, Matrix* b, Matrix* result);
Matrix* matrix_add(Matrix* a, Matrix* b, Matrix* result);
Matrix* matrix_sub(Matrix* a, Matrix* b, Matrix* result);

Matrix* matrix_add_f32(Matrix* a, float b);
Matrix* matrix_sub_f32(Matrix* a, float b);
Matrix* matrix_mul_f32(Matrix* a, float b);
Matrix* matrix_div_f32(Matrix* a, float b);

Matrix* matrix_add_u32(Matrix* a, uint32_t b);
Matrix* matrix_sub_u32(Matrix* a, uint32_t b);
Matrix* matrix_mul_u32(Matrix* a, uint32_t b);
Matrix* matrix_div_u32(Matrix* a, uint32_t b);

Matrix* matrix_add_i32(Matrix* a, int32_t b);
Matrix* matrix_sub_i32(Matrix* a, int32_t b);
Matrix* matrix_mul_i32(Matrix* a, int32_t b);
Matrix* matrix_div_i32(Matrix* a, int32_t b);

Matrix* matrix_transpose(Matrix* m);

void matrix_free(Matrix* m);
