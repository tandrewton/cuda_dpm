# This Makefile assumes the following module files are loaded:
#
# CUDA
#
# This Makefile will only work if executed on a GPU node.
#

NVCC = nvcc

NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets -g

LFLAGS = -lm -Wno-deprecated-gpu-targets -g -lcudart

# Compiler-specific flags (by default, we always use sm_37)
GENCODE_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\"
GENCODE = $(GENCODE_SM37)

.SUFFIXES : .cu .ptx

BINARIES = serialTest, cudaTest

serialTest: serialTest.o
		g++ -O3 --std=c++11 -I src main/test/dpmNVEtest.cpp src/dpm.cpp -o serialTest.o

cudaTest: cudaTest.o
		nvcc -w -O3 -std=c++11 -I src main/test/cudaNVE.cu src/cuda_dpm.cu -o cudaTest.o

clean:
	rm -f *.o $(BINARIES)
