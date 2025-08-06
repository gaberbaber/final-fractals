# executable names
CPU_TARGET = dla_cpu
GPU_TARGET = dla_gpu

# shared source files
PNG_SRC = lodepng.cpp
SUPPORT_SRC = support.cu

# main source files
CPU_MAIN = main_cpu.cu
GPU_MAIN = main_gpu.cu

# default is to build both
all: $(CPU_TARGET) $(GPU_TARGET)

# build CPU version
$(CPU_TARGET): $(CPU_MAIN) $(PNG_SRC) $(SUPPORT_SRC)
	nvcc -O2 -o $(CPU_TARGET) $(CPU_MAIN) $(PNG_SRC) $(SUPPORT_SRC)

# build GPU version
$(GPU_TARGET): $(GPU_MAIN) $(PNG_SRC) $(SUPPORT_SRC)
	nvcc -O2 -o $(GPU_TARGET) $(GPU_MAIN) $(PNG_SRC) $(SUPPORT_SRC)

# clean up
clean:
	rm -f $(CPU_TARGET) $(GPU_TARGET) dla_cpu_*.png dla_gpu_*.png

.PHONY: all clean
