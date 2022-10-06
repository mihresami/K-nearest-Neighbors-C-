#include "cuda_commons.cu"
__global__ void insertion_sort_kernel(int* tags, int dataset_n, int to_predict_n, int k,
                                      float* distances, int* gpu_results) {
    // gpu_results tiene en cada fila los índices de los k más cercanos
    int to_pred_i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    float actual_dist;

    if(to_pred_i >= to_predict_n)
        return;

    gpu_results[to_pred_i * k] = tags[0];

    for (int i = 1; i < dataset_n; i++) {
        actual_dist = distances[to_pred_i * dataset_n + i];

        j = i - 1;

        // No me importa que quede desordenada la parte de la fila depsués de k
        if (i >= k && actual_dist >= distances[to_pred_i * dataset_n + k - 1]) {
            continue;
        }

        while (actual_dist < distances[to_pred_i * dataset_n + j] && j >= 0) {
            distances[to_pred_i * dataset_n + j + 1] = distances[to_pred_i * dataset_n + j];
            if (j + 1 < k) {
                // Sólo copio el tag si estoy en una posición menor que k
                gpu_results[to_pred_i * k + j + 1] = gpu_results[to_pred_i * k + j];
            }
            j--;
        }

        distances[to_pred_i * dataset_n + j + 1] = actual_dist;
        if (j + 1 < k){
            gpu_results[to_pred_i * k + j + 1] = tags[i];
        }
    }
}


__global__ void quick_sort_kernel(int* gpu_tags, int dataset_n, int to_predict_n, int k,
                        float* distances, int left, int right, int to_pred_i,
                        float* aux_distances, int* aux_tags, int* l_size) {
    __shared__ float shared_mem_distances[32];
    __shared__ int shared_mem_tags[32];
    unsigned position, cant_less_pivot;
    float thread_distance;

    if(to_pred_i >= to_predict_n)
        return;

    float pivot_distance = distances[to_pred_i * dataset_n + left];

    // desde donde escribir en el array auxiliar. Va a ir sumando la cantidad de valores que pone
    // cada iteración a la izquierda, porque son menores que el pivot
    int left_from = left;
    // desde donde escribir en el array auxiliar. Va a ir sumando la cantidad de valores que pone
    // cada iteración a la izquierda, porque son menores que el pivot
    int right_to = right;

    // empiezo en 1 porque ya sé que el 0 es el pivot
    for (int i = left + 1; i < right; i+=32) {
        if (i + threadIdx.x >= right) {
            continue;
        }

        thread_distance = distances[to_pred_i * dataset_n + i + threadIdx.x];

        // 32 bits que en la posicion i tienen un 1 si la distancia del thread i del warp es >= al pivot, 0 sino
        unsigned lt_pivot = __ballot_sync(0xffffffff, thread_distance < pivot_distance);
        bool is_lt_pivot = lt_pivot >> (threadIdx.x % 32) & true;
        if(is_lt_pivot) {
            // si el thread tiene una distancia < que el pivot, tiene que contar de izquierda a derecha
            // Cuento cuantos threads anteriores son 1, van a poner su resultado a la izquierda del pivot
            position = __popc(lt_pivot << 32 - threadIdx.x);
        } else {
            // si el thread tiene una distancia >= que el pivot, tiene que contar de derecha a izquierda
            position = 31 - (threadIdx.x - __popc(lt_pivot << 32 - threadIdx.x));
        }
        shared_mem_distances[position] = thread_distance;
        shared_mem_tags[position] = gpu_tags[to_pred_i * dataset_n + i + threadIdx.x];

        cant_less_pivot = __popc(lt_pivot);
        __syncthreads();

        // escritura coalesced, una para los mayores y otra para los menores
        int tid_32 = threadIdx.x % 32;
        if (tid_32 < cant_less_pivot) {
            // Escribo los valores menores al pivot en aux_distances (que va a ser distances el loop que viene)
            // Estos los necesito siempre
            aux_distances[to_pred_i * dataset_n + left_from + tid_32] = shared_mem_distances[tid_32];
            aux_tags[to_pred_i * dataset_n + left_from + tid_32] = shared_mem_tags[tid_32];
        } else if (cant_less_pivot < k) {
            // Si estoy con un valor mayor que el pivot, sólo tengo que escribirlo si el pivot quedó a la izquierda de k
            // si no, ya sé que va a ser mayor que otros k elementos y lo voy a descartar, me ahorro la escritura.
            aux_distances[to_pred_i * dataset_n + right_to - 32 + tid_32 + max(0, i + 32 - right)] = shared_mem_distances[tid_32 + max(0, i + 32 - right)];
            aux_tags[to_pred_i * dataset_n + right_to - 32 + tid_32 + max(0, i + 32 - right)] = shared_mem_tags[tid_32 + max(0, i + 32 - right)];
        }

        left_from += cant_less_pivot;
        right_to -= (32 - cant_less_pivot);
    }

    if (threadIdx.x == 0) {
        aux_distances[to_pred_i * dataset_n + left_from] = pivot_distance;
        aux_tags[to_pred_i * dataset_n + left_from] = gpu_tags[to_pred_i * dataset_n + left];
        *l_size = left_from - left + 1;
    }
}

void quick_sort(int* tags, int dataset_n, int to_predict_n, int k,
                float** distances_gpu, int* gpu_results) {
    // results tiene en cada fila los índices de los k más cercanos
    int *left_size = (int*)malloc(sizeof(int));
    int *left_size_gpu;
    CUDA_CHK(cudaMalloc((void**)&left_size_gpu, sizeof(int)));

    bool *result_is_in_aux = (bool*)malloc(to_predict_n * k * sizeof(bool));

    float *aux_distances_gpu;
    CUDA_CHK(cudaMalloc((void**)&aux_distances_gpu, dataset_n * to_predict_n * sizeof(float)));

    int *aux_tags_gpu;
    CUDA_CHK(cudaMalloc((void**)&aux_tags_gpu, dataset_n * to_predict_n * sizeof(int)));
    int *tags_gpu;
    CUDA_CHK(cudaMalloc((void**)&tags_gpu, dataset_n * to_predict_n * sizeof(int)));

    int block_size_copy = 32;
    dim3 tamBlock_copy(block_size_copy, block_size_copy);
    dim3 tamGrid_copy(dataset_n / block_size_copy, to_predict_n / block_size_copy);
    if (dataset_n % block_size_copy != 0) tamGrid_copy.x += 1;
    if (to_predict_n % block_size_copy != 0) tamGrid_copy.y += 1;

    copy_tags_kernel <<< tamGrid_copy, tamBlock_copy >>> (tags, tags_gpu, dataset_n, to_predict_n);

    dim3 tamBlock_sort(32, 1);
    dim3 tamGrid_sort(1, 1);

    bool even_loops;
    int k_to_find;
    bool *result_is_in_aux_gpu;
    int left;
    int right;

    CUDA_CHK(cudaMalloc((void**)&result_is_in_aux_gpu, to_predict_n * k * sizeof(bool)));

    for (int to_pred_i = 0; to_pred_i < to_predict_n; to_pred_i++) {
        even_loops = false;
        k_to_find = k;
        left = 0;
        right = dataset_n;
        while (k_to_find > 0) {

            quick_sort_kernel <<< tamGrid_sort, tamBlock_sort >>> (
                    tags_gpu, dataset_n, to_predict_n, k_to_find, *distances_gpu, left, right,
                    to_pred_i, aux_distances_gpu, aux_tags_gpu, left_size_gpu);

            // en el array auxiliar están las nuevas distancias, que voya usar para leer
            // el otro tiene lo viejo, lo voy sobreescribir
            swap_arrays(*distances_gpu, aux_distances_gpu);
            swap_arrays(aux_tags_gpu, tags_gpu);

            CUDA_CHK(cudaMemcpy(left_size, left_size_gpu, sizeof(int), cudaMemcpyDeviceToHost));

            if (k_to_find >= *left_size) {
                // el pivot quedó a la izquierda de k, encontré algunos de los más valores más chicos
                k_to_find = k_to_find - *left_size;
                int top = (k < left + *left_size) ? k : left + *left_size;
                for (int i = left; i < top; i++) {
                    result_is_in_aux[to_pred_i * k + i] = !even_loops;
                }
                left += *left_size;
            } else {
                // - 1 porque descarto al pivot
                right = left + *left_size - 1;
            }
            even_loops = !even_loops;
        }
        if (even_loops){
            swap_arrays(*distances_gpu, aux_distances_gpu);
            swap_arrays(aux_tags_gpu, tags_gpu);
            even_loops = !even_loops;
        }
    }

    dim3 tamBlock_results(block_size_copy, block_size_copy);
    dim3 tamGrid_results(k / block_size_copy, to_predict_n / block_size_copy);
    if (k % block_size_copy != 0) tamGrid_results.x += 1;
    if (to_predict_n % block_size_copy != 0) tamGrid_results.y += 1;

    CUDA_CHK(cudaMemcpy(result_is_in_aux_gpu, result_is_in_aux, to_predict_n * k * sizeof(bool), cudaMemcpyHostToDevice));
    copy_results_kernel <<< tamGrid_results, tamBlock_results >>> (
        tags_gpu, aux_tags_gpu, result_is_in_aux_gpu, gpu_results, k, to_predict_n, dataset_n
    );
}


__global__ void quick_sort_kernel_better_pivot(int* gpu_tags, int dataset_n, int to_predict_n, int k,
                        float* distances, int left, int right, int to_pred_i,
                        float* aux_distances, int* aux_tags, int* l_size) {
    __shared__ float shared_mem_distances[32];
    __shared__ int shared_mem_tags[32];
    unsigned position, cant_less_pivot;
    float thread_distance;

    int dist_index = to_pred_i * dataset_n;
    int middle_index = left + ((right - left) / 2);
    float left_dist = distances[dist_index + left];
    float right_dist = distances[dist_index + right - 1];
    float middle_dist = distances[dist_index + middle_index];
    int pivot_index;

    // ^ es xor, elijo como pivot a la mediana del primer valor, el último y el del medio
    if ((left_dist > right_dist) ^ (left_dist > middle_dist)) {
        pivot_index = left;
    } else if ((right_dist < left_dist) ^ (right_dist < middle_dist)) {
        pivot_index = right - 1;
    } else {
        pivot_index = middle_index;
    }
    float pivot_distance = distances[dist_index + pivot_index];

    // desde donde escribir en el array auxiliar. Va a ir sumando la cantidad de valores que pone
    // cada iteración a la izquierda, porque son menores que el pivot
    int left_from = left;
    // desde donde escribir en el array auxiliar. Va a ir sumando la cantidad de valores que pone
    // cada iteración a la izquierda, porque son menores que el pivot
    int right_to = right;

    int i;
    for (int j = left; j < right - 1; j+=32) {
        if (j + threadIdx.x >= pivot_index) {
            i = j+1;
        } else {
            i = j;
        }

        if (i + threadIdx.x >= right) {
            continue;
        }
        thread_distance = distances[dist_index + i + threadIdx.x];

        // 32 bits que en la posicion i tienen un 1 si la distancia del thread i del warp es >= al pivot, 0 sino
        unsigned lt_pivot = __ballot_sync(0xffffffff, thread_distance < pivot_distance);
        bool is_lt_pivot = lt_pivot >> (threadIdx.x % 32) & true;
        if(is_lt_pivot) {
            // si el thread tiene una distancia < que el pivot, tiene que contar de izquierda a derecha
            // Cuento cuantos threads anteriores son 1, van a poner su resultado a la izquierda del pivot
            position = __popc(lt_pivot << 32 - threadIdx.x);
        } else {
            // si el thread tiene una distancia >= que el pivot, tiene que contar de derecha a izquierda
            position = 31 - (threadIdx.x - __popc(lt_pivot << 32 - threadIdx.x));
        }
        shared_mem_distances[position] = thread_distance;
        shared_mem_tags[position] = gpu_tags[dist_index + i + threadIdx.x];

        cant_less_pivot = __popc(lt_pivot);

        __syncthreads();

        // escritura coalesced, una para los mayores y otra para los menores
        // Los threads de la izquierda, del 0 al cant_less_pivot van a escribir los de la izquierda
        // Los thread de la derecha, desde el 32 hasta el último antes de los menores (o del pivot)
        // van a escribir los mayores que el pivot
        if (threadIdx.x < cant_less_pivot) {
            // Escribo los valores menores al pivot en aux_distances (que va a ser distances el loop que viene)
            // Estos los necesito siempre
            aux_distances[dist_index + left_from + threadIdx.x] = shared_mem_distances[threadIdx.x];
            aux_tags[dist_index + left_from + threadIdx.x] = shared_mem_tags[threadIdx.x];
        } else if (cant_less_pivot < k) {
            // Si estoy con un valor mayor que el pivot, sólo tengo que escribirlo si el pivot quedó a la izquierda de k
            // si no, ya sé que va a ser mayor que otros k elementos y lo voy a descartar, me ahorro la escritura.
            int tid = threadIdx.x;
            if (right - i < 32) {
                // No están todos los threads activos, tengo que mover estos a la derecha hasta que el último sea el 31
                tid += 32 - (right - (j + 1));
            }

            aux_distances[dist_index + right_to - (32 - tid)] = shared_mem_distances[tid];
            aux_tags[dist_index + right_to - (32 - tid)] = shared_mem_tags[tid];
        }

        left_from += cant_less_pivot;
        right_to -= (32 - cant_less_pivot);
    }

    if (threadIdx.x == 0) {
        aux_distances[dist_index + left_from] = pivot_distance;
        aux_tags[dist_index + left_from] = gpu_tags[dist_index + pivot_index];

        *l_size = left_from - left + 1;
    }
}


void quick_sort_better_pivot(int* tags, int dataset_n, int to_predict_n, int k,
                float** distances_gpu, int* gpu_results) {
    // results tiene en cada fila los índices de los k más cercanos
    int *left_size = (int*)malloc(sizeof(int));
    int *left_size_gpu;
    CUDA_CHK(cudaMalloc((void**)&left_size_gpu, sizeof(int)));

    bool *result_is_in_aux = (bool*)malloc(to_predict_n * k * sizeof(bool));

    float *aux_distances_gpu;
    CUDA_CHK(cudaMalloc((void**)&aux_distances_gpu, dataset_n * to_predict_n * sizeof(float)));

    int *aux_tags_gpu;
    CUDA_CHK(cudaMalloc((void**)&aux_tags_gpu, dataset_n * to_predict_n * sizeof(int)));
    int *tags_gpu;
    CUDA_CHK(cudaMalloc((void**)&tags_gpu, dataset_n * to_predict_n * sizeof(int)));

    int block_size_copy = 32;
    dim3 tamBlock_copy(block_size_copy, block_size_copy);
    dim3 tamGrid_copy(dataset_n / block_size_copy, to_predict_n / block_size_copy);
    if (dataset_n % block_size_copy != 0) tamGrid_copy.x += 1;
    if (to_predict_n % block_size_copy != 0) tamGrid_copy.y += 1;

    copy_tags_kernel <<< tamGrid_copy, tamBlock_copy >>> (tags, tags_gpu, dataset_n, to_predict_n);

    dim3 tamBlock_sort(32, 1);
    dim3 tamGrid_sort(1, 1);

    bool even_loops;
    bool *result_is_in_aux_gpu;
    int k_to_find, left, right;

    CUDA_CHK(cudaMalloc((void**)&result_is_in_aux_gpu, to_predict_n * k * sizeof(bool)));

    for (int to_pred_i = 0; to_pred_i < to_predict_n; to_pred_i++) {
        even_loops = false;
        k_to_find = k;
        left = 0;
        right = dataset_n;
        while (k_to_find > 0) {

            quick_sort_kernel_better_pivot <<< tamGrid_sort, tamBlock_sort >>> (
                    tags_gpu, dataset_n, to_predict_n, k_to_find, *distances_gpu, left, right,
                    to_pred_i, aux_distances_gpu, aux_tags_gpu, left_size_gpu);

            swap_arrays(*distances_gpu, aux_distances_gpu);
            swap_arrays(aux_tags_gpu, tags_gpu);

            CUDA_CHK(cudaMemcpy(left_size, left_size_gpu, sizeof(int), cudaMemcpyDeviceToHost));

            // - 1 porque descarto al pivot, si el k+1 es el pivot ya sé que los k anteriores son < al pivot
            if (k_to_find >= *left_size - 1) {
                k_to_find = k_to_find - *left_size;
                int top = (k < left + *left_size) ? k : left + *left_size;
                for (int i = left; i < top; i++) {
                    result_is_in_aux[to_pred_i * k + i] = !even_loops;
                }
                left += *left_size;
            } else {
                // - 1 porque descarto al pivot
                right = left + *left_size - 1;
            }
            even_loops = !even_loops;
        }
        if (even_loops){
            swap_arrays(*distances_gpu, aux_distances_gpu);
            swap_arrays(aux_tags_gpu, tags_gpu);
            even_loops = !even_loops;
        }

    }

    dim3 tamBlock_results(block_size_copy, block_size_copy);
    dim3 tamGrid_results(k / block_size_copy, to_predict_n / block_size_copy);
    if (k % block_size_copy != 0) tamGrid_results.x += 1;
    if (to_predict_n % block_size_copy != 0) tamGrid_results.y += 1;

    CUDA_CHK(cudaMemcpy(result_is_in_aux_gpu, result_is_in_aux, to_predict_n * k * sizeof(bool), cudaMemcpyHostToDevice));
    copy_results_kernel <<< tamGrid_results, tamBlock_results >>> (
        tags_gpu, aux_tags_gpu, result_is_in_aux_gpu, gpu_results, k, to_predict_n, dataset_n
    );
}


__global__ void quick_sort_kernel_improved(int* gpu_tags, int dataset_n, int to_predict_n, int k,
                    float* distances, float* aux_distances, int* aux_tags, bool* result_is_in_aux) {
    extern __shared__ float shared_mem[];

    int block_size = min(blockDim.y, to_predict_n);
    float* shared_mem_distances = (float*)shared_mem;
    int*   shared_mem_tags = (int*)&shared_mem_distances[32 * blockDim.y];
    int*   k_to_find = (int*)&shared_mem_tags[32 * blockDim.y];

    int to_pred_i = blockIdx.y * blockDim.y + threadIdx.y;
    if (to_pred_i >= to_predict_n){
        return;
    }

    unsigned position, cant_less_pivot;
    float thread_distance, left_dist, right_dist, middle_dist, pivot_distance;
    int middle_index, pivot_index, left_size;

    // copiar los valores iniciales para todos los to_pred_i
    if (threadIdx.x == 0) {
        if (to_pred_i < to_predict_n) {
            k_to_find[threadIdx.y] = k;
        } else{
            // Estos hilos no van a trabajar, no tienen que encontrar nada. Pero necesito este valor en 0
            // para poder chequear bien que todos los hilos hayan terminado antes de retornar y evitar
            // un deadlock por los syncthreads
            k_to_find[threadIdx.y] = 0;
        }
    }

    int left = 0;
    int right = dataset_n;

    bool everyone_is_finished = false;
    bool even_loops = false;

    int i, left_writing, right_writing;
    int dist_index = to_pred_i * dataset_n;

    while (!everyone_is_finished) {

        left_writing = left;
        right_writing = right;

        // Calculo el pivot
        middle_index = left_writing + ((right_writing - left_writing) / 2);
        left_dist = distances[dist_index + left_writing];
        right_dist = distances[dist_index + right_writing - 1];
        middle_dist = distances[dist_index + middle_index];

        // ^ es xor, elijo como pivot a la mediana del primer valor, el último y el del medio
        if ((left_dist > right_dist) ^ (left_dist > middle_dist)) {
            pivot_index = left_writing;
        } else if ((right_dist < left_dist) ^ (right_dist < middle_dist)) {
            pivot_index = right_writing - 1;
        } else {
            pivot_index = middle_index;
        }
        pivot_distance = distances[dist_index + pivot_index];

        // Iteraciones de 32 en 32 para colocar los puntos de to_pred_i a la izquierda o derecha del pivot
        for (int j = left; j < right - 1; j+=32) {
            // No quiero comparar el pivot
            if (j + threadIdx.x >= pivot_index) {
                i = j+1;
            } else {
                i = j;
            }

            if (i + threadIdx.x < right) {
                thread_distance = distances[dist_index + i + threadIdx.x];
            } else {
                // necesito que cant_less_pivot quede bien para todos los hilos porque lo van a usar para
                // calcular los nuevos límites así que estos valores tienen que ser mayores que el pivot.
                thread_distance = INT_MAX;
            }

            // 32 bits que en la posicion i tienen un 1 si la distancia del thread i del warp es >= al pivot, 0 sino
            unsigned lt_pivot = __ballot_sync(0xffffffff, thread_distance < pivot_distance);
            bool is_lt_pivot = lt_pivot >> (threadIdx.x % 32) & true;
            if (is_lt_pivot) {
                // si el thread tiene una distancia < que el pivot, tiene que contar de izquierda a derecha
                // Cuento cuantos threads anteriores son 1, van a poner su resultado a la izquierda del pivot
                position = __popc(lt_pivot << 32 - threadIdx.x);
                // printf("to_pred %d threadIdx es %d, i es %d, bin %d, gtepivot %u, thread_distance es %.5f es menor. popc es %d position %u\n",
                //        to_pred_i, threadIdx.x, i, thread_distance < pivot_distance, lt_pivot, thread_distance, __popc(lt_pivot << 32 - threadIdx.x), position);
            } else {
                // si el thread tiene una distancia >= que el pivot, tiene que contar de derecha a izquierda
                position = 31 - (threadIdx.x - __popc(lt_pivot << 32 - threadIdx.x));
                // printf("to_pred %d threadIdx es %d, i es %d, bin %d, gtepivot %u, thread_distance es %.5f es mayor. popc es %d position %u\n",
                //        to_pred_i, threadIdx.x, i, thread_distance < pivot_distance, lt_pivot, thread_distance, __popc(lt_pivot << 32 - threadIdx.x), position);
            }

            if (i + threadIdx.x < right) {
                shared_mem_distances[threadIdx.y * 32 + position] = thread_distance;
                shared_mem_tags[threadIdx.y * 32 + position] = gpu_tags[dist_index + i + threadIdx.x];
            }

            cant_less_pivot = __popc(lt_pivot);

            __syncthreads();

            if (i + threadIdx.x < right) {

                // escritura coalesced, una para los mayores y otra para los menores
                if (threadIdx.x < cant_less_pivot) {
                    // Escribo los valores menores al pivot en aux_distances (que va a ser distances el loop que viene)
                    // Estos los necesito siempre

                    aux_distances[dist_index + left_writing + threadIdx.x] = shared_mem_distances[threadIdx.x];
                    aux_tags[dist_index + left_writing + threadIdx.x] = shared_mem_tags[threadIdx.x];
                } else if (left_writing + cant_less_pivot < k) {
                    // Si estoy con un valor mayor que el pivot, sólo tengo que escribirlo si el pivot quedó a la izquierda de k
                    // si no, ya sé que va a ser mayor que otros k elementos y lo voy a descartar, me ahorro la escritura.
                    int tid = threadIdx.x;
                    if (right - i < 32) {
                        // No están todos los threads activos, tengo que mover estos a la derecha hasta que el último sea el 31
                        tid += 32 - (right - (j + 1));
                    }

                    aux_distances[dist_index + right_writing - (32 - tid)] = shared_mem_distances[tid];
                    aux_tags[dist_index + right_writing - (32 - tid)] = shared_mem_tags[tid];
                }

            }
            left_writing += cant_less_pivot;
            right_writing -= (32 - cant_less_pivot);
        }

        // El primer thread de to_pred_i es el encargado de escribir al pivot en aux_distances
        if (threadIdx.x == 0) {
            aux_distances[dist_index + left_writing] = pivot_distance;
            aux_tags[dist_index + left_writing] = gpu_tags[dist_index + pivot_index];
        }

        left_size = left_writing - left + 1;

        if (k_to_find[threadIdx.y] >= left_size - 1) {
            // Guardo la información de si los nuevos resultados están en aux o no
            int top = (k < left + left_size) ? k : left + left_size;

            if (threadIdx.x >= (k - k_to_find[threadIdx.y]) && threadIdx.x < top) {
                result_is_in_aux[to_pred_i * k + threadIdx.x] = !even_loops;
            }
            if (threadIdx.x == 0) {
                k_to_find[threadIdx.y] = k_to_find[threadIdx.y] - left_size;
            }
            left += left_size;
        } else {
            // - 1 porque descarto al pivot
            right = left + left_size - 1;
        }
        even_loops = !even_loops;

        swap_arrays_gpu(distances, aux_distances);
        swap_arrays_gpu(aux_tags, gpu_tags);
        __syncthreads();

        everyone_is_finished = true;
        for(int i=0; i < blockDim.y; i++) {
            if (k_to_find[i] > 0){
                everyone_is_finished = false;
                break;
            }
        }
    }
}


void quick_sort_improved(int* tags, int dataset_n, int to_predict_n, int k,
                float** distances_gpu, int* gpu_results) {

    float *aux_distances_gpu;
    CUDA_CHK(cudaMalloc((void**)&aux_distances_gpu, dataset_n * to_predict_n * sizeof(float)));

    int *aux_tags_gpu;
    CUDA_CHK(cudaMalloc((void**)&aux_tags_gpu, dataset_n * to_predict_n * sizeof(int)));
    int *tags_gpu;
    CUDA_CHK(cudaMalloc((void**)&tags_gpu, dataset_n * to_predict_n * sizeof(int)));

    int block_size = 32;
    dim3 tamBlock(block_size, block_size);
    dim3 tamGrid_copy(dataset_n / block_size, to_predict_n / block_size);
    if (dataset_n % block_size != 0) tamGrid_copy.x += 1;
    if (to_predict_n % block_size != 0) tamGrid_copy.y += 1;

    copy_tags_kernel <<< tamGrid_copy, tamBlock >>> (tags, tags_gpu, dataset_n, to_predict_n);

    bool *result_is_in_aux_gpu;
    CUDA_CHK(cudaMalloc((void**)&result_is_in_aux_gpu, to_predict_n * k * sizeof(bool)));

    int shared_size = 32 * block_size * (sizeof(float) + 2 * sizeof(int));

    block_size = 1;
    dim3 tamBlock_sort(32, block_size);
    dim3 tamGrid_sort(1, to_predict_n / block_size);
    if (to_predict_n % block_size != 0) tamGrid_sort.y += 1;
    quick_sort_kernel_improved <<< tamGrid_sort, tamBlock_sort, shared_size >>> (
            tags_gpu, dataset_n, to_predict_n, k, *distances_gpu,
            aux_distances_gpu, aux_tags_gpu, result_is_in_aux_gpu);
    CUDA_CHK(cudaDeviceSynchronize());

    dim3 tamBlock_results(block_size, block_size);
    dim3 tamGrid_results(k / block_size, to_predict_n / block_size);
    if (k % block_size != 0) tamGrid_results.x += 1;
    if (to_predict_n % block_size != 0) tamGrid_results.y += 1;

    copy_results_kernel <<< tamGrid_results, tamBlock_results >>> (
        tags_gpu, aux_tags_gpu, result_is_in_aux_gpu, gpu_results, k, to_predict_n, dataset_n
    );
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
        gpu_winners[cant_tages+to_pred_i] = winner;
    }
}