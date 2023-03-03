// Author: Ulises Olivares
// uolivares@unam.mx
// Oct 1,2020

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>

#define N 90000000

using namespace std;

void generateRandom(int *h_a);
void parallelAddition();
void serialAddition();
void compareVectors(int *parallelC, int *serialC);

// Variables globales
double serialTimer = 0;
double parallelTimer = 0;

int *h_a, *h_b, *h_c, *serialC;
int *d_a, *d_b, *d_c;
int size = N * sizeof(int);


// Kernel vectorAdd
__global__ void vectorAdd(int *h_a, int *h_b, int *h_c){
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	h_c[index] = h_a[index] + h_b[index];
}

int main(){
	h_a = (int *) malloc(size);
	h_b = (int *) malloc(size);
	h_c = (int *) malloc(size);
	serialC = (int *) malloc(size);


	// initialize arrays with random numbers
	generateRandom(h_a);
	generateRandom(h_b);


	parallelAddition();

	serialAddition();

	compareVectors(h_c, serialC);

	cout << "Speed-up: " << serialTimer / (parallelTimer /1000)<< "X"<<endl; 

	free(h_a); free(h_b); free(h_c); free(serialC);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

	return 0;
}

void parallelAddition(){

	// Reservar memoria en device
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Transferir datos de host h_a device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);


	int blocks = ceil(N / 1024) + 1;
	int threads = 1024;

	// define timers 
	cudaEvent_t start, stop;

	// events to take time
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	// Launch kernel
	vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c);

	cudaEventRecord(stop,0);

	cudaEventSynchronize(stop);

	float parallelTimer = 0;

	cudaEventElapsedTime(&parallelTimer, start, stop);

	cout<< "Elapsed parallel timer: " << parallelTimer << " ms, " << parallelTimer / 1000 << " secs" <<endl;

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
}

void compareVectors(int *parallelC, int *serialC){
	
	int diff = 0;
	for(int i= 0; i<N; i++)
		if(parallelC[i] != serialC[i]){
			diff++;
			cout << "Parallel: " << parallelC[i] << " Serial: " << serialC[i] <<endl; 
		}

	
	if(diff>0){
		cout<< diff <<" elements different" << endl;

	}
	else
		cout << "Vectors are equal!..." << endl;
}

void serialAddition(){
	
	clock_t start = clock();
	for(int i= 0; i<N; i++)
		h_c[i] = h_a[i] + h_b[i];
	clock_t end = clock();

	serialTimer = double (end-start) / double(CLOCKS_PER_SEC);
	cout << "Elapsed time serial: " << serialTimer << endl;
}


void generateRandom(int *h_a){
	
	// Initialize seed
	srand(time(NULL));

	for(int i=0; i<N; i++){
		h_a[i] = rand() % 100 +1;
	}

}
