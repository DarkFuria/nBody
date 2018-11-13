// Small functions which can be used in both versions: CPU and GPU

// All frame data will be saved in this structure
#ifndef HELPERS_H
#define HELPERS_H

typedef struct{
    double * masses;
    double * x;
    double * y;
    double * z;
    double * vx;
    double * vy;
    double * vz;
    double * devMasses;
    double * devX;
    double * devY;
    double * devZ;
    double * devVx;
    double * devVy;
    double * devVz;
} frame;

// function reads frame from csv file
frame * readFrame(char const* frameName);

void * protectedMallocF(char const* arrName, unsigned int size);

// function writes frame into csv file(with velocitys and masses)
void writeFrameFull(char const* frameName, const frame* fr );
// fucntion writes only coordinates into file
void writeFrameShort(char const* frameName, const frame* fr );

// finction prints frame
void printFrame(frame const* fr);

// finction prints square matrix
void printSquareMatrix(const double ** matrix);

// function free's frame
void freeFrame(frame* fr);

// function checks and prints cuda runtime errors
void checkCudaErrors(char const* errMsg);

// allocated array on device and processes errors 
double* cudaProtectedMalloc(char const* arrName, unsigned int size);

// copys array on device and processes errors 
void cudaProtectedMemcpyD(char const* errMsg, double * devPtr, double * hostPtr, unsigned int size);

// copys array from device and processes errors 
void cudaProtectedMemcpyH(char const* errMsg, double * hostPtr, double * devPtr, unsigned int size);

#endif
