CC = nvcc

INCLUDES := -I/home/beomsik/dp/cutlass/include
OBJS := $(wildcard ../tensorop_iwgrad_kernels/*.o)

NVCCFLGAS = # "-gencode=arch=compute_80,code=compute_80"

all: kernel_test_main.cu
	$(CC) $(INCLUDES) $(NVCCFLAGS) $(OBJS) kernel_test_main.cu -o kernel_test_main