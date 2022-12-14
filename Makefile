# This Makefile assumes the following module files are loaded:
#
# CUDA
#
# This Makefile will only work if executed on a GPU node.
# PAT github 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

NVCC = nvcc

NVCCFLAGS = -O3 -Wno-deprecated-gpu-targets -g

LFLAGS = -lm -Wno-deprecated-gpu-targets -lcudart

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

# Compiler-specific flags (by default, we always use sm_37)
GENCODE_SM37 = -gencode=arch=compute_37,code=\"sm_37,compute_37\"
GENCODE_SM60 = -gencode-arch=compute_60,code=\"sm_60,compute_60\"
GENCODE = $(GENCODE_SM37)

.SUFFIXES : .cu .ptx

BINARIES = serialTest, cudaTest

serialTest: serialTest.o
		g++ -O3 --std=c++11 -I src main/test/dpmNVEtest.cpp src/dpm.cpp -o serialTest.o

cudaTest: cudaTest.o
		nvcc -w $(GENCODE_SM37) $(NVCCFLAGS) -std=c++11 -I src main/test/cudaNVE.cu src/cuda_dpm.cu $(LFLAGS) -o cudaTest.o

clean:
	rm -f *.o $(BINARIES)
