// Author: Ulises Olivares
// uolivares@unam.mx
// Oct 22, 2020

#include<iostream>
#include<stdio.h>
#include<time.h>
#include<cstdlib>
#include<math.h>
#include <unistd.h>

#define n 99999999		// input/output 1D array  size 
#define m 9999			//assume mask size as odd

#define TILE_SIZE 1024
#define MAX_MASK_WIDTH 256

using namespace std;

//Global variables
long long  int sizeN = n  * sizeof(float);
long long int sizeM = m * sizeof(float);

float h_N[n] , h_M[m], h_P[n];

int threads = 1024;
int blocks = ceil(float(n)/float(threads));

__constant__ float c_M[m];

// GPU timers using CUDA events
float globalMemTimer = 0.0f, constantMemTimer = 0.0f, sharedMemTimer = 0.0f;



// Method definition
void generateRandom(float *h_a, int size);
void parallelConvolution1D();
void parallelConvolutionConstant1D();
void parallelConvolutionTiled1D();

template <typename vec>
void printVector(vec *V, int size);

// Kernel definition
__global__ void CUDAConvolution1D(float *N, float *M, float *P, int Mask_Width, int  Width);
__global__ void CUDAConvolutionConstant1D(float *N, float *P, int Mask_Width, int  Width);
__global__ void CUDAconvolution_1D_tiled(float *N, float *P, int Mask_Width, int Width);

int main(){


	//init N and M with random numbers
	generateRandom(h_N, n);
	generateRandom(h_M, m);

	// Parallel convolution 1D kernel
	parallelConvolution1D();

	// Parallel convolution 1D constant memory
	parallelConvolutionConstant1D();

	// Parallel convolution 1D shared - constant memory
	parallelConvolutionTiled1D();
	
	return 0;
}

__global__ void CUDAConvolution1D(float *N, float *M, float *P, int Mask_Width, int  Width){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float Pvalue = 0;
	int N_start_point = i - (Mask_Width/2);

	for (int j = 0; j < Mask_Width; j++) {
		if (N_start_point + j >= 0 && N_start_point + j < Width) {
			Pvalue += N[N_start_point + j]*M[j];
		}
	}
	P[i] = Pvalue;
}

__global__ void CUDAConvolutionConstant1D(float *N, float *P,  int Mask_Width, int  Width){
	
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("M[i]: %d  ", c_M[i]  );
	//printf("thread: %d", i );
	float Pvalue = 0;
	int N_start_point = i - (Mask_Width/2);

	for (int j = 0; j < Mask_Width; j++) {
		if (N_start_point + j >= 0 && N_start_point + j < Width) {
			Pvalue += N[N_start_point + j]*c_M[j];
		}
	}
	P[i] = Pvalue;
}


__global__ void CUDAconvolution_1D_tiled(float *N, float *P, int Mask_Width, int Width) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("tid: %d ", i);
	__shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];
	int n1 = Mask_Width/2;
	int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;

	if (threadIdx.x >= blockDim.x - n1) {
		N_ds[threadIdx.x - (blockDim.x - n1)] =
		(halo_index_left < 0) ? 0 : N[halo_index_left];
	}
	
	N_ds[n1 + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x];
	int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
	
	if (threadIdx.x < n1) {
		N_ds[n1 + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? 0 : N[halo_index_right];
	}
	__syncthreads();
	float Pvalue = 0;

	for(int j = 0; j < Mask_Width; j++) {
		Pvalue += N_ds[threadIdx.x + j]*c_M[j];
	}
	/*if(Pvalue!=0)	
		printf("value: %f", Pvalue);*/
	P[i] = Pvalue;
	//printf("tid %d Pvalue: %lf ", i, Pvalue );
}


template <typename vec>
void printVector(vec *V, int size){
	for(int i = 0; i < size; i++){
		cout<< V[i] << " ";
	}
	cout << endl;
}

void generateRandom(float *h_a, int size){
	// Initialize seed
	srand(time(NULL));

	for(int i=0; i<size; i++){
		h_a[i] = float(rand() % 10 +1);
	}
}

void parallelConvolutionTiled1D() {
	float *d_N, *d_P;
	
	cudaMalloc((void **)&d_N, sizeN);
	cudaMalloc((void **)&d_P, sizeN);

	// copy data from host to device
	cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);
	
	// Trasfeer data to constant memory
	cudaMemcpyToSymbol(c_M, h_M, sizeM);

	// define timers 
	cudaEvent_t start, stop;

	// events to take time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start timer
	cudaEventRecord(start,0);

	//Launch kernel
	CUDAconvolution_1D_tiled<<<blocks, threads>>> (d_N, d_P, m, n);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&sharedMemTimer, start, stop);
	cudaDeviceSynchronize();

	cout<< "Elapsed parallel 1D convolution (Shared-Constant Mem) : " << sharedMemTimer << " ms, " << sharedMemTimer / 1000 << " secs" <<endl;

	cudaMemcpy(h_P, d_P, sizeN, cudaMemcpyDeviceToHost);

	//printVector(h_P, n);

	cudaFree(c_M); cudaFree(d_N); cudaFree(d_P);

}

void parallelConvolutionConstant1D(){
	float *d_N, *d_P;
	
	cudaMalloc((void **)&d_N, sizeN);
	cudaMalloc((void **)&d_P, sizeN);

	// copy data from host to device
	cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);
	
	// Trasfeer data to constant memory
	cudaMemcpyToSymbol(c_M, h_M, sizeM);

	// define timers 
	cudaEvent_t start, stop;

	// events to take time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start timer
	cudaEventRecord(start,0);

	//Launch kernel
	CUDAConvolutionConstant1D<<<blocks, threads>>>(d_N, d_P, m, n);

	cudaEventRecord(stop,0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&constantMemTimer, start, stop);

	cudaDeviceSynchronize();

	cout<< "Elapsed parallel 1D convolution (Constant Mem) : " << constantMemTimer << " ms, " << constantMemTimer / 1000 << " secs" <<endl;

	cudaMemcpy(h_P, d_P, sizeN, cudaMemcpyDeviceToHost);
	//cout<< "Resulting P vector (Constant)" << endl;
	//printVector(h_P, n);

	cudaFree(c_M); cudaFree(d_N); cudaFree(d_P);
}

void parallelConvolution1D(){
	float *d_N, *d_M, *d_P;
	// Reservar memoria en device
	cudaMalloc((void **)&d_N, sizeN);
	cudaMalloc((void **)&d_M, sizeM);
	cudaMalloc((void **)&d_P, sizeN);

	// Transferir datos de host a device
	cudaMemcpy(d_N, h_N, sizeN, cudaMemcpyHostToDevice);
	cudaMemcpy(d_M, h_M, sizeM, cudaMemcpyHostToDevice);

	
	// define timers 
	cudaEvent_t start, stop;

	// events to take time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start timer
	cudaEventRecord(start,0);

	//Launch kernel
	CUDAConvolution1D<<<blocks, threads>>>(d_N, d_M, d_P, m, n);

	cudaEventRecord(stop,0);

	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&globalMemTimer, start, stop);

	//cudaDeviceSynchronize();

	cout<< "Elapsed parallel 1D convolution (Global Mem) : " << globalMemTimer << " ms, " << globalMemTimer / 1000 << " secs" <<endl;

	cudaMemcpy(h_P, d_P, sizeN, cudaMemcpyDeviceToHost);

	//cout<< "Resulting P vector (Global)" << endl;

	//printVector(h_P, n);

	//free(h_N); free(h_M); free(h_P); 
	cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);

}





