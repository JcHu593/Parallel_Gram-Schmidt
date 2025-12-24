CC = gcc
CXX = g++
CXXFLAGS = -O3 -fopenmp -march=native -lm
NVCC = nvcc
HIPCC = hipcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx908

LDFLAGS = -lm

EXES = gs-seq gs-cuda bgs-cuda


.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

gs-seq: gs-seq.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

gs-cuda: gs-cuda.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

bgs-cuda: bgs-cuda.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?


