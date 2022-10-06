#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__global__ void copy_tags_kernel(int* tags, int* gpu_tags, int dataset_n, int to_predict_n) {
    int dataset_i = blockIdx.x * blockDim.x + threadIdx.x;
    int to_pred_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (dataset_i >= dataset_n || to_pred_i >= to_predict_n)
        return;

    gpu_tags[to_pred_i * dataset_n + dataset_i] = tags[dataset_i];
}


__global__ void copy_results_kernel(int* tags_gpu, int* aux_tags_gpu, bool* result_is_in_aux_gpu, int* gpu_results, int k, int to_predict_n, int dataset_n) {
    int k_i = blockIdx.x * blockDim.x + threadIdx.x;
    int to_pred_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (k_i >= k || to_pred_i >= to_predict_n)
        return;

    int index = to_pred_i * k + k_i;
    if (result_is_in_aux_gpu[index]){
        gpu_results[index] = aux_tags_gpu[to_pred_i * dataset_n + k_i];
    } else {
        gpu_results[index] = tags_gpu[to_pred_i * dataset_n + k_i];
    }
}

void swap_arrays(int* &p1, int* &p2){
    int* tmp = p1;
    p1 = p2;
    p2 = tmp;
}
void swap_arrays(float* &p1, float* &p2){
    float* tmp = p1;
    p1 = p2;
    p2 = tmp;
}

__device__ void swap_arrays_gpu(int* &p1, int* &p2){
    int* tmp = p1;
    p1 = p2;
    p2 = tmp;
}

__device__ void swap_arrays_gpu(float* &p1, float* &p2){
    float* tmp = p1;
    p1 = p2;
    p2 = tmp;
}


__global__ void count_winner_kernel(int* k_results, int* gpu_winners, int to_predict_n, int k, int cant_tags) {
    extern __shared__ int shared_mem_count[];

    int* shared_winners = shared_mem_count;
    int* shared_k_results = (int*)&shared_mem_count[cant_tags];

    // Voy a usarlo para dos cosas, es el índice que indica qué columna cargar en shared memory
    // Y también es qué tag contar cuántas veces está entre los primeros k tags
    int to_count_i = blockIdx.x * blockDim.x + threadIdx.x;

    int to_pred_i = blockIdx.y * blockDim.y + threadIdx.y;

    if(to_pred_i < to_predict_n){
        // Cargo el bloque de resultados (los tags de los k mas cercanos para cada ejemplo a predecir)
        // a la memoria compartida
        if (to_count_i < k) {
            shared_k_results[threadIdx.y * k + threadIdx.x] = k_results[to_pred_i * k + to_count_i];
        }
    }
    __syncthreads();

    if (to_pred_i < to_predict_n && to_count_i < cant_tags){
        int counter = 0;
        for (int i = 0; i < k; i++) {
            if (shared_k_results[threadIdx.y * k + i] == to_count_i)
                counter += 1;
        }
        shared_winners[threadIdx.y * cant_tags + threadIdx.x] = counter;
    }
    __syncthreads();

    if (to_pred_i < to_predict_n && to_count_i == 0) {
        int winner = 0;
        int max = -1;

        for (int tag = 0; tag < cant_tags; tag++) {
            if (shared_winners[threadIdx.y * cant_tags + tag] > max) {
                max = shared_winners[threadIdx.y * cant_tags + tag];
                winner = tag;
            }
        }
        gpu_winners[to_pred_i] = winner;
    }
}