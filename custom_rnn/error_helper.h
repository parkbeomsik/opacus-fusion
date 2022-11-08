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
      assert(0);                                                        \
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

