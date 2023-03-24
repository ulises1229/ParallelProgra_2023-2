#include <iostream>
#include <omp.h>
#include <vector>
#include <stdlib.h>
#include <chrono>


using namespace std;
using namespace chrono;

#define N 900000000

// Definición de funciones
void parallelSum();
void serialSum();
void initVectors();
int equalVectors();

// Definición de variables globales
vector<int> A(N);
vector<int> B(N);
vector<int> serialC(N);
vector<int> parallelC(N);

int main() {
    // Inicializar vectores con números aleatorios
    auto start = high_resolution_clock::now();
    initVectors();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "Initialization finished, elapsed time: "<<  duration.count()/1000 << " milliseconds" <<endl;

 // Realizar suma serial
 start = high_resolution_clock::now();
 serialSum();
 end = high_resolution_clock::now();
 auto serialTimer = duration_cast<microseconds>(end - start);
 cout << "Serial sum finished, elapsed time: "<<  serialTimer.count()/1000 << " milliseconds" <<endl;


 // Realizar suma paralela
 start = high_resolution_clock::now();
 parallelSum();
 end = high_resolution_clock::now();
 auto parallelTimer = duration_cast<microseconds>(end - start);
 cout << "Parallel sum finished, elapsed time: "<<  parallelTimer.count()/1000 << " milliseconds" <<endl;


 // Calcular speedUp
 cout << "SpeedUp = " << float(serialTimer.count()/parallelTimer.count()) << endl;

 if (equalVectors() == 0)
     cout << "Los vectores son distintos" << endl;
 else
     cout << "Los vectores son iguales"<< endl;


 return 0;
}


/*
* Function to initialize two vectors A and B randomly.
*/
void initVectors(){
    //#pragma omp parallel for
    for(int i=0; i<N; i++){
        A[i] =  i * 2; // Rnd Nums 1 - 100
        B[i] = i * 31; // Rnd Nums 1 - 100
        //A[i] =  rand() % 100 + 1 ; // Rnd Nums 1 - 100
        //B[i] =  rand() % 100 + 1 ; // Rnd Nums 1 - 100
    }
}

void serialSum(){
    for(int i= 0; i<N; i++){
        serialC[i] = A[i] + B[i];
    }
}

void parallelSum(){
    #pragma omp parallel
    cout << "Número de hilos: "<< omp_get_num_threads << endl;
    #pragma omp for
    for(int i = 0; i<N; i++){
        parallelC[i] = A[i] + B[i];
    }
}

int equalVectors(){
    int equal = 1;
    for(int i  =0; i<N; i++){
        if(serialC[i] != parallelC[i]){
            equal = 0;
            break;
        }
    }
    return equal;
}