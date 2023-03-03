#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <array>
#include <bits/stdc++.h>

using namespace std;

#define M 5
#define N 4

// Method definition

void randomArrays();
void printArrays();
void printResultArray();
void mergeSequential();
int coRank(int k, int *A, int m, int *B, int n);

array<int, M> A;
array<int, N> B;
array<int, M+N> C;

int main() {

    // Init arrays randomly
    srand(time(NULL));
    randomArrays();

    // Print arrays
    cout << "Unsorted arrays" << endl;
    printArrays();

    // Sort arrays
    sort(A.begin(), A.end());
    sort(B.begin(), B.end());

    cout << "Sorted arrays" << endl;
    printArrays();

    mergeSequential();

    cout << "Single sorted array" << endl;
    printResultArray();


    return 0;
}

void printResultArray() {
    // Init first array
    for (int i = 0; i < M+N; i++) {
        cout << C[i] << " \t";
    }
}

void printArrays(){
    // Init first array
    for(int i= 0; i<M; i++){
        cout << A[i] << " \t";
    }
    cout << endl;
    // Init second array
    for(int i= 0; i<M; i++){
        cout << B[i] << " \t";
    }
    cout << endl;
}

void randomArrays(){
    // Init first array
    for(int i= 0; i<M; i++){
        A[i] = (rand() % 10) + 1;
    }
    // Init second array
    for(int i= 0; i<M; i++){
        B[i] = (rand() % 10) + 1;
    }
}

__device__ void mergeSequential() {
    int i = 0; //index into A
    int j = 0; //index into B
    int k = 0; //index into C
    // handle the start of A[] and B[]
    while ((i < M) && (j < N)) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == M) {
        //done with A[] handle remaining B[]
        for (; j < N; j++) {
            C[k++] = B[j];
        }
    } else {
        //done with B[], handle remaining A[]
        for (; i <M; i++) {
            C[k++] = A[i];
        }
    }
}

int coRank(int k, int *A, int m, int *B, int n) {
    int i= k<m ? k : m; //i = min(k,m)
    int j = k- i;
    int i_low = 0>(k-n) ? 0 : k-n; //i_low = max(0, k-n)
    int j_low = 0>(k-m) ? 0 : k-m; //i_low = max(0, k-m)
    int delta;
    bool active = true;
    while(active) {
        if (i > 0 && j < n && A[i-1] > B[j]) {
            delta = ((i - i_low +1) >> 1); // ceil(i-i_low)/2)
            j_low = j;
            j = j + delta;

            i = i - delta;
        } else if (j > 0 && i < m && B[j-1] >= A[i]) {
            delta = ((j - j_low +1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}


__global__ void merge_basic_kernel(int *A, int m, int *B, int n, int *C)
{
    __device__ float ceilf (float x);
    int tid= blockIdx.x * blockDim.x + threadIdx.x;
    int k_curr = tid * ceilf((m+n) / (blockDim.x * gridDim.x); // start index of output
    int k_next = min((tid+1) * ceil((m+n) / (blockDim.x*gridDim.x)), m+n); // end index of output
    int i_curr= co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr -i_curr;
    int j_next = k_next-i_next;
    //All threads call the sequential merge function
    mergeSequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr] );
}

__global__ void merge_tiled_kernel(int* A, int m, int* B, int n, int* C, int tile_size)
{
    // shared memory allocation
    extern _shared_ int shareAB[];
    int * A_S = &shareAB[0]; //shareA is first half of shareAB
    int * B_S = &shareAB[tile_size]; //ShareB is second half of ShareAB
    int C_curr = blockIdx.x * ceil((m+n)/gridDim.x) ; // starting point of the C subarray for current block
    int C_next = min((blockIdx.x+1) * ceil((m+n)/gridDim.x), (m+n)); // starting point for next block
    if (threadIdx.x ==0)º
    {
        A_S[0] = co_rank(C_curr, A, m, B, n); // Make the block-level co-rank values visible to
        A_S[1] = co_rank(C_next, A, m, B, n); // other threads in the block
    }
    _syncthreads();
    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    _syncthreads();

    int counter = 0; //iteration counter
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil((C_length)/tile_size); //total iteration
    int C_completed = 0;
    int A_consumed = 0;

    int B_consumed = 0;
    while(counter < total_iteration)
    {
        // loading tile-size A and B elements into shared memory
        for(int i=0; i<tile_size; i+=blockDim.x)
        {
            if( i + threadIdx.x < A_length - A_consumed)
            {
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x ];
            }
        }
        for(int i=0; i<tile_size; i+=blockDim.x)
        {
            if(i + threadIdx.x < B_length - B_consumed)
            {
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size/blockDim.x);
        int c_next = (threadIdx.x+1) * (tile_size/blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
        // find co-rank for c_curr and c_next
        int a_curr = co_rank(c_curr, A_S, min(tile_size, A_length-A_consumed),
                             B_S, min(tile_size, B_length-B_consumed));
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_S, min(tile_size, A_length-A_consumed),
                             B_S, min(tile_size, B_length-B_consumed));
        int b_next = c_next - a_next;
        // All threads call the sequential merge function
        mergeSequential (A_S+a_curr, a_next-a_curr, B_S+b_curr, b_next-b_curr,
                          C+C_curr+C_completed+c_curr);
        // Update the A and B elements that have been consumed thus far
        counter ++;
        C_completed += tile_size;
        A_consumed += co_rank(tile_size, A_S, tile_size, B_S, tile_size);
        B_consumed = C_completed - A_consumed;
        _syncthreads();
    }
}
