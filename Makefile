# ------------ config ------------
NVCC        = nvcc
# one of these might be right
SM_ARCHS    = -gencode arch=compute_70,code=sm_70 \
              -gencode arch=compute_80,code=sm_80 \
              -gencode arch=compute_86,code=sm_86 \
              -gencode arch=compute_89,code=sm_89
NVCC_FLAGS  = -O3 -std=c++14 -I/usr/local/cuda/include $(SM_ARCHS)

# Runtime/linker flags 
LD_FLAGS_COMMON = -lcudart -L/usr/local/cuda/lib64
LD_FLAGS_GPU    = $(LD_FLAGS_COMMON) -lcurand

# ------------ targets ------------
CPU_TARGET = dla_cpu
GPU_TARGET = dla_gpu

# ------------ sources ------------
PNG_SRC     = lodepng.cpp
SUPPORT_SRC = support.cu
CPU_MAIN    = main_cpu.cu
GPU_MAIN    = main_gpu.cu
KERNEL_SRC  = kernel.cu

# Objects
CPU_OBJ = main_cpu.o lodepng.o support.o
GPU_OBJ = main_gpu.o kernel.o  lodepng.o support.o

# Default: build both
all: $(CPU_TARGET) $(GPU_TARGET)

# ------------ build rules ------------
# Compile CUDA sources to .o (both .cu and .cpp through nvcc)
%.o: %.cu
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

%.o: %.cpp
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

# Link CPU version (still links cudart because support.cu uses CUDA timing)
$(CPU_TARGET): $(CPU_OBJ)
	$(NVCC) $(CPU_OBJ) -o $@ $(LD_FLAGS_COMMON)

# Link GPU version (needs curand too)
$(GPU_TARGET): $(GPU_OBJ)
	$(NVCC) $(GPU_OBJ) -o $@ $(LD_FLAGS_GPU)

# ------------ housekeeping ------------
clean:
	rm -f *.o $(CPU_TARGET) $(GPU_TARGET) dla_cpu_*.png dla_gpu_*.png

.PHONY: all clean
