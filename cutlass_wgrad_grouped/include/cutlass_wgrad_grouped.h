#pragma once

#include <vector>
#include <string>
// #include "base_operation.h"
// #include "cutlass_error.h"

namespace cutlass_wgrad_grouped {

/// Status code returned by CUTLASS operations
enum class Status {
    kSuccess,                    ///< Operation was successful.
    kErrorMisalignedOperand,     ///< operands fail alignment requirements.
    kErrorInvalidDataType,       ///< DataType fails requirement.
    kErrorInvalidLayout,         ///< Layout fails alignment requirement.
    kErrorInvalidProblem,        ///< Specified problem size is not supported by operator.
    kErrorNotSupported,          ///< Operation is not supported on current device.
    kErrorWorkspaceNull,         ///< The given workspace is null when it is required to be non-null.
    kErrorInternal,              ///< An error within CUTLASS occurred.
    kErrorArchMismatch,          ///< CUTLASS runs on a device that it was not compiled for.
    kErrorInsufficientDriver,    ///< CUTLASS runs with a driver that is too old.
    kErrorMemoryAllocation,      ///< Kernel launch failed due to insufficient device memory.
    kInvalid                     ///< Status is unspecified.
};

/// Convert cutlass status to status strings
static char const* cutlassGetStatusString(Status status) {
    switch (status) {
        case Status::kSuccess:
        return "Success";
        case Status::kErrorMisalignedOperand:
        return "Error Misaligned Operand";
        case Status::kErrorInvalidDataType:
        return "Error Invalid Data Type";
        case Status::kErrorInvalidLayout:
        return "Error Invalid Layout";
        case Status::kErrorInvalidProblem:
        return "Error Invalid Problem";
        case Status::kErrorNotSupported:
        return "Error Not Supported";
        case Status::kErrorWorkspaceNull:
        return "Error Workspace Null";
        case Status::kErrorInternal:
        return "Error Internal";
        case Status::kErrorInsufficientDriver:
        return "Error Insufficient Driver";
        case Status::kErrorArchMismatch:
        return "Error Architecture Mismatch";
        case Status::kErrorMemoryAllocation:
        return "Error Memory Allocation failed";
        case Status::kInvalid: break;
    }

    return "Invalid status";
}

struct Conv2dConfig {
    int N;
    int H;
    int W;
    int C;
    int K;
    int R;
    int S;
    int P;
    int Q;
    int pad_h;
    int pad_w;
    int stride_h;
    int stride_w;
    int dilation_h;
    int dilation_w;

    int split_k_slices;
};

// Base class for all operations
class Operation {
public:
  std::string name;

  Operation(std::string name) {this->name = name;};
  virtual ~Operation() { }

  virtual uint64_t get_host_workspace_size() const = 0;

  virtual size_t get_workspace_size(
    void * _args, 
    void *host_workspace) const = 0;
  
  virtual Status initialize(
    void const *_arguments,
    void *device_workspace, 
    void *host_workspace) const = 0;

  virtual Status run(void *host_workspace) const = 0;

  virtual Status update_ptrs(
    void **ptr_A,
    void **ptr_B,
    void **ptr_C,
    void **ptr_D,
    int problem_count,
    void *host_workspace) const = 0;
};


struct OperationWithWorkspace {
    Operation * operation;
    void * host_workspace;
};



extern void * _device_problems;

extern std::vector<Operation *> operations;
extern std::vector <void *> device_workspaces;
// extern std::vector <void *> host_workspaces;

extern std::vector<OperationWithWorkspace> operations_with_workspaces;

extern void ** device_ptr_A;
extern void ** device_ptr_B;
extern void ** device_ptr_C;
extern void ** device_ptr_D;

#if defined(_USE_TENSOR_CORE)
void initialize_int_tensorop();
#else
void initialize_int();
#endif
void initialize_float();

// size_t get_device_problem_size_in_bytes (int problem_count);
template<typename dType>
void initialize_problems(std::vector<Conv2dConfig> const &);

OperationWithWorkspace get_best_operation(void ** ptr_A,
                              void ** ptr_B,
                              void ** ptr_C,
                              void ** ptr_D);

Status run(OperationWithWorkspace);
Status update_ptrs(OperationWithWorkspace,
                   void ** ptr_A,
                   void ** ptr_B,
                   void ** ptr_C,
                   void ** ptr_D,
                   int problem_count);

void finalize();

} // namespace cutlass_wgrad_grouped