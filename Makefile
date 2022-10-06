CC=gcc
CXX=g++
NVCC=nvcc

NVCCFLAGS=-w -O3 -std=c++11

knn-exec: main.cu knnCPU.cpp
	$(NVCC) $(NVCCFLAGS) main.cu sort.cu cuda_commons.cu knnCPU.cpp -o knn-exec

clean:
	rm -f knn-exec
