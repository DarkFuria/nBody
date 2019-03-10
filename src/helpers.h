// Small functions which can be used in both versions: CPU and GPU

// All frame data will be saved in this structure
#ifndef HELPERS_H
#define HELPERS_H

#include <cuda_runtime.h>

typedef struct{
    float4 * bodys;
	float4 * vels;
    float4 * accels;
    float4 * devBodys;
	float4 * devVels;
	float4 * devAccels;
} frame;

// prints info about program
void printHelp();

// allocates memory with all needed checks
void *protectedMalloc(char const* arrName, size_t size);

// function reads frame from csv file
frame * readFrame(char const* frameName, int N_BODYS);

// function writes frame into csv file(with velocitys and masses)
void writeFrameFull(char const* frameName, const frame* fr, int N_BODYS);
// fucntion writes only coordinates into file
void writeFrameShort(char const* frameName, const frame* fr, int N_BODYS);

// finction prints frame
void printFrame(frame const* fr, int N_BODYS);

// function free's frame
void freeFrame(frame* fr);

// function checks and prints cuda runtime errors
void checkCudaErrors(char const* errMsg);

// allocated array on device and processes errors 
void* cudaProtectedMalloc(char const* arrName, unsigned int size);

// copys array on device and processes errors 
void cudaProtectedMemcpyD(char const* errMsg, void * devPtr, void * hostPtr, unsigned int size);

// copys array from device and processes errors 
void cudaProtectedMemcpyH(char const* errMsg, void * hostPtr, void * devPtr, unsigned int size);

#endif
