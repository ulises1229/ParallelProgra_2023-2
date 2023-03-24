// Author: Ulises Olivares
// uolivares@unam.mx
// Oct 22, 2020

#include<iostream>
#include<stdio.h>
#include<time.h>
#include<cstdlib>
#include<math.h>

#define n 900000
#define m 10000

using namespace std;

//Global variables
long long  int sizeN = n  * sizeof(float);
long long int sizeM = m * sizeof(float);

float h_N[n] , h_M[m], h_P[n];

int threads = 512;
int blocks = ceil(float(n)/float(threads));

__constant__ float c_M[m];

// GPU timers using CUDA events
float globalMemTimer = 0, constantMemTimer = 0;



// Method definition
void generateRandom(float *h_a, int size);
void parallelConvolution1D();
void parallelConvolutionConstant1D();

template <typename vec>
void printVector(vec *V, int size);

__global__ void CUDAConvolution1D(float *N, float *M, float *P, int Mask_Width, int  Width);

__global__ void CUDAConvolutionConstant1D(float *N, float *P, int Mask_Width, int  Width);


int main(){


	//init N and M with random numbers
	generateRandom(h_N, n);
	generateRandom(h_M, m);

	// Parallel convolution 1D kernel
	parallelConvolution1D();

	// Parallel convolution 1D constant memory
	parallelConvolutionConstant1D();
	
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

	cout<< "Elapsed parallel 1D convolution (Constant Mem) : " << constantMemTimer << " ms, " << globalMemTimer / 1000 << " secs" <<endl;

	cudaMemcpy(h_P, d_P, sizeN, cudaMemcpyDeviceToHost);
	//cout<< "Resulting P vector (Constant)" << endl;
	//printVector(h_P, n);
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




