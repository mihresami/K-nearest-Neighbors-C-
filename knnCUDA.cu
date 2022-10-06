#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include "sort.cu"
#include "cuda_commons.cu"
void check_error(cudaError_t err, const char *msg);
void knnParallel(float* coords, float* newCoords, int* classes, int numClasses, int numSamples, int numNewSamples, int k);

__device__ float manhattan_distance_gpu(float x, float y) {
    return fabsf(x - y);
}



__global__ void distances_kernel_naive(float* dataset, float* to_predict, int dataset_n, int dimension,
                           int to_predict_n, float* distances, int distance_algorithm) {
    // Cada hilo en x, y guarda la distancia entre el vector x del dataset y el vector
    // y a predecir
    // distances tiene filas de to_predict_n de ancho
    // cada fila tiene todas las distancias para el to_pred_i contra todos los del dastaset
    int dataset_i = blockIdx.x * blockDim.x + threadIdx.x;
    int to_pred_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (dataset_i >= dataset_n || to_pred_i >= to_predict_n)
        return;

    float distance = 0;

    for (int i = 0; i < dimension ; i++) {
            distance += manhattan_distance_gpu(
                to_predict[to_pred_i * dimension + i],
                dataset[dataset_i * dimension + i]
            );
    }
    distances[to_pred_i * dataset_n + dataset_i] = distance;
}
void knnParallel(float* coords, float* newCoords, int* classes, int numClasses, int numSamples, int numNewSamples, int k) {
    //*** Device-variables-declaration ***
    float* d_coords;
    float* d_newCoords;
    int* d_classes;
    float *distance;
    int totalSamples = numSamples + numNewSamples;
    int *gpu_tags, *gpu_results, *gpu_winners;
    //*** device-allocation ***
    check_error(cudaMalloc(&d_coords, totalSamples * DIMENSION * sizeof(float)), "alloc d_coords_x");
    check_error(cudaMalloc(&d_classes, totalSamples * sizeof(int)), "alloc d_classes");
    check_error(cudaMalloc(&d_newCoords, numNewSamples * DIMENSION * sizeof(float)), "alloc d_coordsnew");
    CUDA_CHK(cudaMalloc((void**)&distance, numSamples * numNewSamples * sizeof(float)));
    //***copy-arrays-on-device***
    check_error(cudaMemcpy(d_coords, coords, totalSamples * DIMENSION * sizeof(float), cudaMemcpyHostToDevice), "copy d_coords");
    check_error(cudaMemcpy(d_classes, classes, totalSamples * sizeof(int), cudaMemcpyHostToDevice), "copy d_classes");
    check_error(cudaMemcpy(d_newCoords, newCoords, numNewSamples * DIMENSION * sizeof(float), cudaMemcpyHostToDevice), "copy d_coordsnew");
    CUDA_CHK(cudaMalloc((void**)&gpu_results, k * numNewSamples * sizeof(int)));
    CUDA_CHK(cudaMalloc((void**)&gpu_winners, totalSamples * sizeof(int)));
    // TODO: Put your parallel code in this function
    /*
       1. Design the KNN parallel code.
       1. Specify the sizes of grid and block.
       2. Launch the kernel function (Write kernel code in knnCUDA.cu).
    */
    dim3 tamGrid, tamBlock;
    int block_size=32;
    tamGrid = dim3(numSamples / block_size, numNewSamples / block_size);
    tamBlock = dim3(block_size, block_size);
    if (numSamples % block_size != 0) tamGrid.x += 1;
    if (numNewSamples % block_size != 0) tamGrid.y += 1;

    distances_kernel_naive <<< tamGrid, tamBlock >>> (
            coords, newCoords, numSamples, 2, numNewSamples, distance
        );
    quick_sort(classes, numSamples, numNewSamples, k, &distance, gpu_results);
    int *results = (int*)malloc(k * numNewSamples * sizeof(int));
    CUDA_CHK(cudaMemcpy(results, gpu_results, k * numNewSamples * sizeof(int), cudaMemcpyDeviceToHost));
    int count_grid_width = k < numSamples ? numSamples : k;
    dim3 tamGrid_count(count_grid_width / block_size, numNewSamples / block_size);
    if (count_grid_width % block_size != 0) tamGrid_count.x += 1;
    if (numNewSamples % block_size != 0) tamGrid_count.y += 1;
    dim3 tamBlock_count(block_size, block_size);
    int shared_size = (k * numNewSamples + numSamples) * sizeof(int);
    count_winner_kernel <<< tamGrid_count, tamBlock_count, shared_size >>> (gpu_results, gpu_winners, numNewSamples, k, numSamples);
    check_error(cudaMemcpy(d_classes, gpu_winners, numNewSamples * sizeof(int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    // download device -> host
    check_error(cudaMemcpy(coords, d_coords, DIMENSION * totalSamples * sizeof(float), cudaMemcpyDeviceToHost), "download coords");
    check_error(cudaMemcpy(classes, d_classes, totalSamples * sizeof(int), cudaMemcpyDeviceToHost), "download classes");
   
}

void check_error(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) 
    {
        fprintf(stderr, "%s : error %d (%s)\n", msg, err, cudaGetErrorString(err));
        exit(err);
    }
}
