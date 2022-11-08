#include "cuda_runtime.h"
#include <iostream>

int main(void) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("%d.%d\n", deviceProp.major, deviceProp.minor);
}