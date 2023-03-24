//
// Created by Ulises Olivares on 27/02/23.
//

#ifndef OPENMPTEST_HISTOGRAM_H
#define OPENMPTEST_HISTOGRAM_H
#define N 10000000
#include <vector>
using namespace std;

class histogram {
public:
    void initVector();
    void parallelHistogram();
    void serialHistogram();

private:
    vector<int> A[N];
    vector<int> hist[10];
};


#endif //OPENMPTEST_HISTOGRAM_H
