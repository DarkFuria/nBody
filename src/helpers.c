#include "helpers.h"

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "settings.h"

frame * readFrame(char const* frameName){
    FILE *inp;
    inp = fopen(frameName, "r");
    if(inp == NULL){
        fprintf(stderr, "ERROR: Can't open file %s\n", frameName);
        exit(1);
    };
    
    frame *tmp = malloc(sizeof(frame));
    if(tmp == NULL){
        fprintf(stderr, "ERROR: Couldnt allocate memory for new frame\n");
        fclose(inp);
        exit(1);
    };
    
	tmp->bodys = malloc(sizeof(float4) * N_BODYS);
	if(tmp->bodys == NULL){
		fprintf(stderr, "ERROR: Couldn't allocate memory for tmp->bodys\n");
		exit(1);
	};
	
	tmp->vels = malloc(sizeof(float3) * N_BODYS);
	if(tmp->vels == NULL){
		fprintf(stderr, "ERROR: Couldn't allocate memory for tmp->vels\n");
		exit(1);
	};
                    
                    
    for(int i = 0; i < N_BODYS; i++){
        if(fscanf(inp, "%E %E %E %E %E %E %E",&tmp->bodys[i].w, &tmp->bodys[i].x, &tmp->bodys[i].y, &tmp->bodys[i].z, &tmp->vels[i].x, &tmp->vels[i].y, &tmp->vels[i].z) != 7) {
            fprintf(stderr, "ERROR: Can't read file %s\n", frameName);
            free(tmp->bodys);
            free(tmp->vels);
            free(tmp);
            fclose(inp);
            exit(1);
        };
    };
    
    fclose(inp);
    return tmp;
};

void printFrame(frame const* fr){
    for(int i = 0; i < N_BODYS; i++){
        fprintf(stdout, "%f %f %f %f %f %f\n", fr->bodys[i].x, fr->bodys[i].y, fr->bodys[i].z, fr->vels[i].x, fr->vels[i].y, fr->vels[i].z);
    };
};

void printSquareMatrix(const double ** matrix){
    for(int i = 0; i < N_BODYS; i++){
        for(int j = 0; j < N_BODYS; j++){
            fprintf(stdout, "%15.2f", matrix[i][j]);
        };
        fprintf(stdout, "\n");
    };
};

void writeFrameFull(char const* frameName,const frame* fr ){
    FILE * out = fopen(frameName, "w");
    if(out == NULL){
        fprintf(stderr, "ERROR: Can't open file %s\n", frameName);
        exit(1);
    };
    for(int i = 0; i < N_BODYS; i++){
        fprintf(out, "%f %f %f %f %f %f %f\n", fr->bodys[i].w, fr->bodys[i].x, fr->bodys[i].y, fr->bodys[i].z, fr->vels[i].x, fr->vels[i].y, fr->vels[i].z);
    };
    fclose(out);
};

void writeFrameShort(char const* frameName,const frame* fr ){
    FILE * out = fopen(frameName, "w");
    if(out == NULL){
        fprintf(stderr, "ERROR: Can't open file %s\n", frameName);
        exit(1);
    };
    for(int i = 0; i < N_BODYS; i++){
        fprintf(out, "%f %f %f\n", fr->bodys[i].x, fr->bodys[i].y, fr->bodys[i].z);
    };
    fclose(out);
};

void freeFrame(frame* fr){
	free(fr->bodys);
	free(fr->vels);
};

void checkCudaErrors(char const* errMsg){
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
		fprintf(stderr, "%s\n", errMsg);
		exit(1);
	};
};

void* cudaProtectedMalloc(char const* arrName, unsigned int size){
	void * tmp;
	cudaMalloc(&tmp, size);
	checkCudaErrors(arrName);
	return tmp;
};

void cudaProtectedMemcpyD(char const* errMsg, void * devPtr, void * hostPtr, unsigned int size){
	cudaMemcpy(devPtr, hostPtr, size, cudaMemcpyHostToDevice);
	checkCudaErrors(errMsg);
};

void cudaProtectedMemcpyH(char const* errMsg, void * hostPtr, void * devPtr, unsigned int size){
	cudaMemcpy(hostPtr, devPtr, size, cudaMemcpyDeviceToHost);
	checkCudaErrors(errMsg);
};
