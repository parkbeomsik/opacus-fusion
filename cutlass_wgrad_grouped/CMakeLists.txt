cmake_minimum_required(VERSION 3.19)
project(cutlass_wgrad_grouped 
        VERSION 1.0.0 
        DESCRIPTION "CUTLASS wgrad impligit gemm grouped"
        LANGUAGES CXX CUDA)

if(NOT DEFINED CUTLASS_PATH)
    message(FATAL_ERROR "Please set CUTLASS_PATH (e.g. -DCUTLASS_PATH=/path/to/cutlass)")
elseif(NOT EXISTS ${CUTLASS_PATH})
    message(FATAL_ERROR "Invalid CUTLASS_PATH")
elseif(NOT IS_DIRECTORY ${CUTLASS_PATH})
    message(FATAL_ERROR "Invalid CUTLASS_PATH")
endif()

include(GNUInstallDirs)

set(CMAKE_INSTALL_LIBDIR ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_BINARY_DIR}/include)

# Find all kernel files in kernel/
if(DEFINED NVCC_ARCHS AND NVCC_ARCHS EQUAL 80)
FILE(GLOB KERNELS CONFIGURE_DEPENDS src/kernel/*.cu src/kernel_int_tensorop/*.cu)
else()
FILE(GLOB KERNELS CONFIGURE_DEPENDS src/kernel/*.cu src/kernel_int/*.cu)
endif()
# set(KERNELS src/cutlass_simt_swgrad_grouped_optimized_32x32x4_16x16x4_1x1x1_3_nhwc.cu)

# For A100
if(DEFINED NVCC_ARCHS AND NVCC_ARCHS EQUAL 80)
add_library(cutlass_wgrad_grouped STATIC
    ${KERNELS}
    ${PROJECT_SOURCE_DIR}/src/initialize_swgrad_grouped.cpp
    # ${PROJECT_SOURCE_DIR}/src/initialize_iwgrad_grouped.cpp
    ${PROJECT_SOURCE_DIR}/src/initialize_iwgrad_tensorop_grouped.cpp
    ${PROJECT_SOURCE_DIR}/src/cutlass_wgrad_grouped.cu
)
# For V100
else()
add_library(cutlass_wgrad_grouped STATIC
    ${KERNELS}
    ${PROJECT_SOURCE_DIR}/src/initialize_swgrad_grouped.cpp
    ${PROJECT_SOURCE_DIR}/src/initialize_iwgrad_grouped.cpp
    ${PROJECT_SOURCE_DIR}/src/cutlass_wgrad_grouped.cu
)
endif()

set_property(TARGET cutlass_wgrad_grouped PROPERTY POSITION_INDEPENDENT_CODE ON)

set_target_properties(cutlass_wgrad_grouped PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    PUBLIC_HEADER include/cutlass_wgrad_grouped.h)

configure_file(cutlass_wgrad_grouped.pc.in cutlass_wgrad_grouped.pc @ONLY)

target_include_directories(cutlass_wgrad_grouped PRIVATE
                           ${PROJECT_SOURCE_DIR}/include
                           ${PROJECT_SOURCE_DIR}/src
                           ${CUTLASS_PATH}/include
)                    

if(DEFINED NVCC_ARCHS AND NVCC_ARCHS EQUAL 80)
message(STATUS "Compute capability =80 is used")
target_compile_options(cutlass_wgrad_grouped PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_80,code=compute_80>)
# target_compile_options(cutlass_wgrad_grouped PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g)
# target_compile_options(cutlass_wgrad_grouped PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G) 
target_compile_definitions(cutlass_wgrad_grouped PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:_USE_TENSOR_CORE>) 
target_compile_definitions(cutlass_wgrad_grouped PRIVATE $<$<COMPILE_LANGUAGE:CXX>:_USE_TENSOR_CORE>) 
else()
message(STATUS "Compute capability <80 is used")
endif()
# target_compile_options(cutlass_wgrad_grouped PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g>)
# target_compile_options(cutlass_wgrad_grouped PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>) 

install(TARGETS cutlass_wgrad_grouped
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# install(FILES ${CMAKE_BINARY_DIR}/cutlass_wgrad_grouped.pc
#     DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

