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
	
	res->distX = cudaProtectedMalloc("tempData->distX", sizeof(double) * N_BODYS * N_BODYS);
	res->distY = cudaProtectedMalloc("tempData->distY", sizeof(double) * N_BODYS * N_BODYS);
	res->distZ = cudaProtectedMalloc("tempData->distZ", sizeof(double) * N_BODYS * N_BODYS);
	res->len = cudaProtectedMalloc("tempData->len", sizeof(double) * N_BODYS * N_BODYS);
	
	res->interX = cudaProtectedMalloc("tempData->interX", sizeof(double) * N_BODYS * N_BODYS);
	res->interY = cudaProtectedMalloc("tempData->interY", sizeof(double) * N_BODYS * N_BODYS);
	res->interZ = cudaProtectedMalloc("tempData->interZ", sizeof(double) * N_BODYS * N_BODYS);
	res->interTotal = cudaProtectedMalloc("tempData->interTotal", sizeof(double) * N_BODYS * N_BODYS);
	
	res->TFX = cudaProtectedMalloc("tempData->TFX", sizeof(double) * N_BODYS);
	res->TFY = cudaProtectedMalloc("tempData->TFY", sizeof(double) * N_BODYS);
	res->TFZ = cudaProtectedMalloc("tempData->TFZ", sizeof(double) * N_BODYS);
	
	res->altX = cudaProtectedMalloc("tempData->altX", sizeof(double) * N_BODYS);
	res->altY = cudaProtectedMalloc("tempData->altY", sizeof(double) * N_BODYS);
	res->altZ = cudaProtectedMalloc("tempData->altZ", sizeof(double) * N_BODYS);
	return res;
};

// frees temporary data structure
void freeTempData(tempData* td){
	cudaFree(td->distX);
	cudaFree(td->distY);
	cudaFree(td->distZ);
	cudaFree(td->len);
	cudaFree(td->interX);
	cudaFree(td->interY);
	cudaFree(td->interZ);
	cudaFree(td->interTotal);
	cudaFree(td->TFX);
	cudaFree(td->TFY);
	cudaFree(td->TFZ);
	cudaFree(td->altX);
	cudaFree(td->altY);
	cudaFree(td->altZ);
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
__global__ void gpu_calculateDistArray(const double *x, const double *y, const double *z, double *distX, double *distY, double *distZ, double *len){
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadID < N_BODYS * N_BODYS){
        int i = threadID % N_BODYS;
        int j = threadID / N_BODYS;
        double dx = (fabs(x[j] - x[i]) < EPSILON) ? EPSILON : x[j] - x[i];
        double dy = (fabs(y[j] - y[i]) < EPSILON) ? EPSILON : y[j] - y[i];
        double dz = (fabs(z[j] - z[i]) < EPSILON) ? EPSILON : z[j] - z[i];
        double l = sqrt(dx * dx + dy * dy + dz * dz);
        distX[threadID] = dx;
        distY[threadID] = dy;
        distZ[threadID] = dz;
        len[threadID] = l;
    };
};

// calculates interactions using Gravitation Law
__global__ void gpu_calculateInteraction(double * __restrict__ fx, double * fy, double * fz, const double *__restrict__ gravitationalParameters, const double *distX, const double * distY, const double * distZ, const double * len){
    unsigned int threadID = blockDim.y * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    if(threadID < N_BODYS * N_BODYS){
        double GP = gravitationalParameters[threadID];
        double r = len[threadID];
        double f = GP / (r * r);
        fx[threadID] = f * distX[threadID] / fabs(r);
        fy[threadID] = f * distY[threadID] / fabs(r);
        fz[threadID] = f * distZ[threadID] / fabs(r);
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

// updates frame
void gpu_updateFrame(frame* fr, double * __restrict__ gravitationalParameters, const tempData* __restrict__ td){
	dim3 matrDim = {(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, (N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, 1};
	dim3 linDim = {(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, 1, 1};
	dim3 blockSize = {THREADS_AMOUNT, 1, 1};
	
	gpu_calculateDistArray<<<matrDim, blockSize >>>(fr->devX, fr->devY, fr->devZ, td->distX, td->distY, td->distZ, td->len);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_CalculateDistArray");
	
	gpu_calculateInteraction<<<matrDim, blockSize>>>(td->TFX, td->TFY, td->TFZ, gravitationalParameters, td->distX, td->distY, td->distZ, td->len);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateInteraction");
	
	// calculating total forces
	gpu_calculateTotalForces<<< matrDim, blockSize>>>(td->TFX, td->interX);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateTotalForces X");
	
	gpu_calculateTotalForces<<< matrDim, blockSize>>>(td->TFY, td->interY);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateTotalForces Y");
	
	gpu_calculateTotalForces<<< matrDim, blockSize>>>(td->TFZ, td->interZ);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateTotalForces Z");
	
	// caltulating alterations
	gpu_calculateAlteration<<<linDim, blockSize>>>(td->altX, td->TFX, fr->devMasses);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateAlteration X");
	
	gpu_calculateAlteration<<<linDim, blockSize>>>(td->altY, td->TFY, fr->devMasses);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateAlteration Y");
	
	gpu_calculateAlteration<<<linDim, blockSize>>>(td->altZ, td->TFZ, fr->devMasses);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_calculateAlteration Z");
	
	// calculating velocities
	gpu_integrate<<<linDim, blockSize>>>(fr->devVx, td->altX);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_integrate velsX");
	
	gpu_integrate<<<linDim, blockSize>>>(fr->devVy, td->altY);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_integrate velsY");
	
	gpu_integrate<<<linDim, blockSize>>>(fr->devVz, td->altZ);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_integrate velsZ");
	
	// updating coordinates
	gpu_integrate<<<linDim, blockSize>>>(fr->devX, fr->devVx);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_integrate coordsX");
	
	gpu_integrate<<<linDim, blockSize>>>(fr->devY, fr->devVy);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_integrate coordsY");
	
	gpu_integrate<<<linDim, blockSize>>>(fr->devZ, fr->devVz);
	cudaDeviceSynchronize();
	checkCudaErrors("gpu_integrate coordsZ");
	
	cudaDeviceSynchronize();
};

