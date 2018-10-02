#include "gpuModel.cuh"
#include "cuda_runtime.h"
extern "C"{
    #include "helpers.h"
    #include "settings.h"
    #include <stdio.h>
}

// creates temporary data structure 
tempData* createTempData(){
	tempData* res = (tempData*)malloc(sizeof(tempData));
	res->dist = cudaProtectedMalloc("tempData->dist", sizeof(double) * N_BODYS * N_BODYS);
	res->interactions = cudaProtectedMalloc("tempData->interactions", sizeof(double) * N_BODYS * N_BODYS);
	res->TF = cudaProtectedMalloc("tempData->TF", sizeof(double) * N_BODYS);
	res->alters = cudaProtectedMalloc("tempData->alters", sizeof(double) * N_BODYS);
	return res;
};

// frees temporary data structure
void freeTempData(tempData* td){
	cudaFree(td->dist);
	cudaFree(td->interactions);
	cudaFree(td->TF);
	cudaFree(td->alters);
};


// takes masses array and returns it into predefined array(N*N)
__global__ void gpu_prepareGravitationalParameters(double* __restrict__ gravitationalParameters, const double* __restrict__ masses){
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadID < N_BODYS * N_BODYS - 1){
        int i = threadID % N_BODYS;
        int j = threadID / N_BODYS;
        gravitationalParameters[threadID] = (i == j) ? 0 : G * masses[i] * masses[j]; // global memory read is very expensive
    };
};

// calculates distance between all points int projection on the one axis
__global__ void gpu_calculateDistArray(double* __restrict__ dist, double* __restrict__ coordinates){
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadID < N_BODYS * N_BODYS){
        int i = threadID % N_BODYS;
        int j = threadID / N_BODYS;
        if(fabs(coordinates[j] - coordinates[i]) > EPSILON){
            dist[threadID] = coordinates[j] - coordinates[i]; 
        } else {
            dist[threadID] = EPSILON;
        };
    };
};

// calculates interactions using Gravitation Law
__global__ void gpu_calculateInteraction(double * __restrict__ forces, const double *__restrict__ gravitationalParameters, const double * __restrict__ dist){
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadID < N_BODYS * N_BODYS){
        double GP = gravitationalParameters[threadID];
        double r = dist[threadID];
        forces[threadID] = GP * r / (r * r * fabs(r));
    };
};

__global__ void gpu_calculateTotalForces(double * __restrict__ totalForces, double * __restrict__ forces){
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadID < N_BODYS * N_BODYS){
        int i = threadID % N_BODYS;
        int j = threadID / N_BODYS;
        if(j == 0){
            totalForces[i] = 0;
        };
        __syncthreads();
        atomicAdd(&totalForces[i], forces[threadID]);
    };
};

// calculates alterations
__global__ void gpu_calculateAlteration(double * alteration, double * __restrict__ totalForces, double * __restrict__ masses){
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadID < N_BODYS){
		alteration[threadID] = totalForces[threadID] / masses[threadID];
	};
};

// integrates using Eulers method
__global__ void gpu_integrate(double * x, double * __restrict__ dx){
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if(threadID < N_BODYS){
		x[threadID] += DELTA_T * dx[threadID];
	};
};

// updates coordinates
void gpu_updateCoordinates(double * coord, double * vels, const double * __restrict__ gravitationalParameters, double * __restrict__ masses, const tempData* __restrict__ td){
	
	
	dim3 matrDim = {(N_BODYS * N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, 1, 1};
	dim3 linDim = {(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, 1, 1};
	dim3 blockSize = {THREADS_AMOUNT, 1, 1};
	
	gpu_calculateDistArray<<<matrDim, blockSize >>>(td->dist, coord);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_CalculateDistArray");
	
	gpu_calculateInteraction<<<matrDim, blockSize>>>(td->interactions, gravitationalParameters, td->dist);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateInteraction");
	
	gpu_calculateTotalForces<<< matrDim, blockSize>>>(td->TF, td->interactions);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateTotalForces");
	
	gpu_calculateAlteration<<<linDim, blockSize>>>(td->alters, td->TF, masses);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateAlteration");
	
	gpu_integrate<<<linDim, blockSize>>>(vels, td->alters);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_integrate vels");
	
	gpu_integrate<<<linDim, blockSize>>>(coord, vels);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_integrate coords");
	
	cudaDeviceSynchronize();
};

// updates frame
void gpu_updateFrame(frame* fr, double * __restrict__ gravitationalParameters, const tempData* __restrict__ td){
	gpu_updateCoordinates(fr->devX, fr->devVx, gravitationalParameters, fr->devMasses, td);
	gpu_updateCoordinates(fr->devY, fr->devVy, gravitationalParameters, fr->devMasses, td);
	gpu_updateCoordinates(fr->devZ, fr->devVz, gravitationalParameters, fr->devMasses, td);
};

