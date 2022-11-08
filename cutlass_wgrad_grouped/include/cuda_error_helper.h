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

#include <stdio.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include "cutlass_wgrad_grouped.h"


#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status                            \
      << "(" << cudaGetErrorString(status) << ")";                    \
      FatalError(_error.str());                                        \
    }                                                                   \
}

#define checkCUTLASS(status) {                                      \
    std::stringstream _error;                                          \
    if (status != cutlass_wgrad_grouped::Status::kSuccess) {             \
      _error << "CUTLASS failure: " << static_cast<int>(status)             \
      << "(" << cutlass_wgrad_grouped::cutlassGetStatusString(status) << ")";                    \
      FatalError(_error.str());                                        \
    }                                                                   \
}
