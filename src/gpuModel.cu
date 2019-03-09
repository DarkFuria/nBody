#include "gpuModel.cuh"
#include "cuda_runtime.h"
extern "C"{
    #include "helpers.h"
    #include "settings.h"
    #include <stdio.h>
}

__device__ float3 gpu_calculateBodyBodyInteraction(float4 bodyI, float4 bodyJ, float3 accel){
	// calculating distances [8 FLOPS]
	float4 dist;
	dist.x = bodyJ.x - bodyI.x;
	dist.y = bodyJ.y - bodyI.y;
	dist.z = bodyJ.z - bodyI.z;
	dist.w = sqrtf(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z + EPSILON * EPSILON);
	
	// calculating alteration [3 FLOPS]
	float a = bodyJ.w * G / (dist.w * dist.w);
	
	// updating acceleration [9 FLOPS]
	accel.x += a * dist.x / dist.w;
	accel.y += a * dist.y / dist.w;
	accel.z += a * dist.z / dist.w;
	return accel;
};

__device__ float3 gpu_submatrixProcessing(float4 updatingBody, float3 bodyAcceleration){
	int i;
	extern __shared__ float4 submatrix[];
	for(i = 0; i < blockDim.x; i+=4){
		bodyAcceleration = gpu_calculateBodyBodyInteraction(updatingBody, submatrix[i], bodyAcceleration);
		bodyAcceleration = gpu_calculateBodyBodyInteraction(updatingBody, submatrix[i+1], bodyAcceleration);
		bodyAcceleration = gpu_calculateBodyBodyInteraction(updatingBody, submatrix[i+2], bodyAcceleration);
		bodyAcceleration = gpu_calculateBodyBodyInteraction(updatingBody, submatrix[i+3], bodyAcceleration);
	};
	return bodyAcceleration;
};


__global__ void gpu_calculateAccelerations(float4* bodys, float4* accels, int N_BODYS){
	extern __shared__ float4 shared[];
	float4 body; // body for updating by this thread
	float3 acceleration = {0.0f, 0.0f, 0.0f};
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	body = bodys[threadID];
	
	for(int i = 0, tile = 0; i < N_BODYS; i+= blockDim.x, tile++){
		int idx = tile * blockDim.x + threadIdx.x;
		shared[threadIdx.x] = bodys[idx];
		__syncthreads();
		acceleration = gpu_submatrixProcessing(body, acceleration);
		__syncthreads();
	};
	
	float4 res = {acceleration.x, acceleration.y, acceleration.z, 0.0f};
	accels[threadID] = res;
};

__global__ void gpu_updateCoordinatesEuler(float4* bodys, float4* vels, float4* accels, float dt){
	int threadID = blockDim.x *blockIdx.x + threadIdx.x;
	float4 body = bodys[threadID];
	float4 vel = vels[threadID];
	float4 acc = accels[threadID];
	
	// updating vel
	vel.x += acc.x * dt;
	vel.y += acc.y * dt;
	vel.z += acc.z * dt;
	
	// updating coordinates
	body.x += vel.x * dt;
	body.y += vel.y * dt;
	body.z += vel.z * dt;
	
	// saving result
	bodys[threadID] = body;
	vels[threadID] = vel;
	
};

__global__ void gpu_integrateEuler(float4 *x, float4 *dx, float dt){
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    float4 curX = x[threadID];
    float4 curDx = dx[threadID];

    curX.x += curDx.x * dt;
    curX.y += curDx.y * dt;
    curX.z += curDx.z * dt;

    x[threadID] = curX;
};

void gpu_updateCoordinatesVelocityVerlet(float4 * coords, float4 * vels, float4 * accels, float dt, int N_BODYS, int THREADS_AMOUNT){
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, dt / 2.0);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, dt);
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, dt / 2.0);
};

void gpu_updateCoordinatesForestRuth(float4 * coords, float4 * vels, float4 * accels, float dt, int N_BODYS, int THREADS_AMOUNT){
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, FR_THETA * dt * 0.5f);
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, FR_THETA * dt);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, (1 - FR_THETA) * dt * 0.5f);
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, (1 - 2.0f * FR_THETA) * dt);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, (1 - FR_THETA) * dt * 0.5f);
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, FR_THETA * dt);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, FR_THETA * dt * 0.5f);
};