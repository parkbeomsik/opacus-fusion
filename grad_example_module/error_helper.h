#pragma once

#include <stdio.h>
#include <string.h>

#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#include <cstdlib>    //C++
#include <cstdio>

#include <iostream>
#include <vector>
#include <deque>

#include <cublas_v2.h>
// #include <cudnn.h>
#include <cudnn.h>
//#include <cnmem.h>
// #include "/home/mrhu/vdnn/cnmem/include/cnmem.h"
// #include "/home/beomsik/vdnn/cnmem/include/cnmem.h"

#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <math.h>

#include <chrono> 
#include <string>

#include <cublas_v2.h>
#include <curand.h>
#include <unistd.h>

#include "cutlass_wgrad_grouped.h"

#include "cutlass/cutlass.h"

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCUDNN(status) {                                           \
    std::stringstream _error;                                           \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << status                            \
      << "(" << cudnnGetErrorString(status) << ")";                    \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCUBLAS(status) {                                          \
    std::stringstream _error;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                              \
      _error << "CUBLAS failure: " << status;                           \
      FatalError(_error.str());                                        \
    }                                                                  \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status                            \
      << "(" << cudaGetErrorString(status) << ")";                    \
      FatalError(_error.str());                                        \
    }                                                                   \
}

#define checkCutlass(status) {                                      \
    std::stringstream _error;                                          \
    if (status != cutlass_wgrad_grouped::Status::kSuccess) {            \
      _error << "Cutlass failure: "                                    \
      << "(" << cutlass_wgrad_grouped::cutlassGetStatusString(status) << ")";  \
      FatalError(_error.str());                                        \
    }                                                                   \
}

#define checkCutlassRaw(status) {                                      \
    std::stringstream _error;                                          \
    if (status != cutlass::Status::kSuccess) {            \
      _error << "Cutlass failure: "                                    \
      << "(" << cutlass::cutlassGetStatusString(status) << ")";  \
      FatalError(_error.str());                                        \
    }                                                                   \
}

// cuRAND API errors
static const char *curandGetErrorString(curandStatus_t error)
{
    switch (error)
    {
        case CURAND_STATUS_SUCCESS:
            return "CURAND_STATUS_SUCCESS";

        case CURAND_STATUS_VERSION_MISMATCH:
            return "CURAND_STATUS_VERSION_MISMATCH";

        case CURAND_STATUS_NOT_INITIALIZED:
            return "CURAND_STATUS_NOT_INITIALIZED";

        case CURAND_STATUS_ALLOCATION_FAILED:
            return "CURAND_STATUS_ALLOCATION_FAILED";

        case CURAND_STATUS_TYPE_ERROR:
            return "CURAND_STATUS_TYPE_ERROR";

        case CURAND_STATUS_OUT_OF_RANGE:
            return "CURAND_STATUS_OUT_OF_RANGE";

        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

        case CURAND_STATUS_LAUNCH_FAILURE:
            return "CURAND_STATUS_LAUNCH_FAILURE";

        case CURAND_STATUS_PREEXISTING_FAILURE:
            return "CURAND_STATUS_PREEXISTING_FAILURE";

        case CURAND_STATUS_INITIALIZATION_FAILED:
            return "CURAND_STATUS_INITIALIZATION_FAILED";

        case CURAND_STATUS_ARCH_MISMATCH:
            return "CURAND_STATUS_ARCH_MISMATCH";

        case CURAND_STATUS_INTERNAL_ERROR:
            return "CURAND_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define checkCurand(status) {                                      \
    std::stringstream _error;                                          \
    if (status != CURAND_STATUS_SUCCESS) {                             \
      _error << "Curand failure: " << status                            \
      << "(" << curandGetErrorString(status) << ")";                    \
      FatalError(_error.str());                                        \
    }                                                                   \
}
