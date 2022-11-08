CC = nvcc

INCLUDES := -I/home/beomsik/dp/cutlass/include


NVCCFLGAS = # "-gencode=arch=compute_80,code=compute_80"
NVCCFLAGS = "-gencode=arch=compute_70,code=compute_70"

iwgrad_tensorop: test_iwgrad_tensorop.cu
	OBJS := $(wildcard ../_tensorop_iwgrad_kernels/*.o)
	
	$(CC) $(INCLUDES) $(NVCCFLAGS) $(OBJS) test_iwgrad_tensorop.cu -o test_iwgrad_tensorop

iwgrad_tensorop: test_iwgrad_simt.cu
	OBJS := $(wildcard ../_simt_iwgrad_kernels/*.o)
	
	$(CC) $(INCLUDES) $(NVCCFLAGS) $(OBJS) test_iwgrad_simt.cu -o test_iwgrad_simt

igemm_batched_simt: test_igemm_batched_simt.cu
	OBJS := $(wildcard ../_simt_igemm_batched_kernels/*.o)
	
	$(CC) $(INCLUDES) $(NVCCFLAGS) $(OBJS) test_igemm_batched_simt.cu -o test_igemm_batched_simt