#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

// global variable definition
// GPU timers using CUDA events
float unified = 0.0f, traditional = 0.0f, unified_initD = 0.0f;

// CUDA kernel to add elements of two arrays
__global__ void add(int N, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;
    //for (int i = index; i < n; i += stride)
    if (index <N ) {
        y[index] = x[index] + y[index];
        //printf("y Value, %f\n", y[index]);
    }
}

__global__ void init(int N, float *x, float*y) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    //int stride = blockDim.x * gridDim.x;
    //for(int i = index; i < n; i +=stride){
    if (index < N) {
        x[index] = 1.0f;
        y[index] = 2.0f;
    }
    //}
}

__global__ void print(int N, float *x, float*y){
    printf("Hello from gpu!...\n");
}

void unifiedVectorAdd(){
    int N = 999999;
    float *x, *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Launch kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);


    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    //cout<< " Vector Add (Unified memory) : " << unified << " ms, " << unified / 1000 << " secs" <<endl;

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
}



void unifiedInitDevice(){
    int N = 999999;
    cout << "N: " << N << endl;
    float *x, *y;
    // define timers
   /* cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timer
    cudaEventRecord(start,0);*/

    // Allocate Unified Memory -- accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // Launch kernel to initialize input arrays

    // Launch kernel on 1M elements on the GPU
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;
    cout << "Threads: " <<threads << endl;
    cout << "blocks: " << blocks << endl;


    init<<<threads, blocks>>>(N, x, y);
    add<<<threads, blocks>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();




    //cudaDeviceSynchronize();

    //cout<< " Vector Add (Unified memory init in device) : " << unified << " ms, " << unified / 1000 << " secs" <<endl;

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError += fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
}

int main(void)
{
    //unifiedVectorAdd();
    unifiedInitDevice();

    return 0;
}
