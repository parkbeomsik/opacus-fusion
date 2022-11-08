#pragma once


void compute_scaling_factor_cuda(float *out, 
                                const float *norm,
                                float max_norm,
                                 int num_rows_to_compute=1);

void compute_scaling_factor2_cuda(float *out, 
                                const float *norm,
                                const float *norm2,
                                float max_norm);                              