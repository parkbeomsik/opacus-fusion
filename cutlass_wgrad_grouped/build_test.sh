# nvcc -I/home/beomsik/dp/cutlass/include -I./include -I./src src/cutlass_simt_swgrad_grouped_optimized_32x32x4_16x16x4_1x1x1_3_nhwc.cu src/cutlass_wgrad_grouped.cu src/initialize_all.cpp test.cpp -o test

cd build
make install -j 80
cd ..
nvcc -L./build/lib -I./build/include -lcutlass_wgrad_grouped -g -G -D_USE_TENSOR_CORE test.cpp -o test