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

__device__ float3 gpu_submatrixProcessing(float4 updatingBody, float3 bodyAcceleration, int N_BODYS){
    int i;
    extern __shared__ float4 submatrix[];
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    for(i = 0; i < blockDim.x && threadID < N_BODYS; i++){
        bodyAcceleration = gpu_calculateBodyBodyInteraction(updatingBody, submatrix[i], bodyAcceleration);
    };
    return bodyAcceleration;
};


__global__ void gpu_calculateAccelerations(float4* bodys, float4* accels, int N_BODYS){
    extern __shared__ float4 shared[];
    float4 body; // body for updating by this thread
    float3 acceleration = {0.0f, 0.0f, 0.0f};
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadID < N_BODYS){
        body = bodys[threadID];
        
        for(int i = 0, tile = 0; i < N_BODYS; i+= blockDim.x, tile++){
            int idx = tile * blockDim.x + threadIdx.x;
            shared[threadIdx.x] = bodys[idx];
            __syncthreads();
            acceleration = gpu_submatrixProcessing(body, acceleration, N_BODYS);
            __syncthreads();
        };
        
        float4 res = {acceleration.x, acceleration.y, acceleration.z, 0.0f};
        accels[threadID] = res;
    };
};

__global__ void gpu_updateCoordinatesEuler(float4* bodys, float4* vels, float4* accels, float dt, int N_BODYS){
    int threadID = blockDim.x *blockIdx.x + threadIdx.x;
    if(threadID < N_BODYS){
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
};

__global__ void gpu_integrateEuler(float4 *x, float4 *dx, float dt, int N_BODYS){
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    if(threadID < N_BODYS){
        float4 curX = x[threadID];
        float4 curDx = dx[threadID];

        curX.x += curDx.x * dt;
        curX.y += curDx.y * dt;
        curX.z += curDx.z * dt;

        x[threadID] = curX;
    };
};

void gpu_updateCoordinatesVelocityVerlet(float4 * coords, float4 * vels, float4 * accels, float dt, int N_BODYS, int THREADS_AMOUNT){
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, dt / 2.0, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, dt, N_BODYS);
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, dt / 2.0, N_BODYS);
};

void gpu_updateCoordinatesForestRuth(float4 * coords, float4 * vels, float4 * accels, float dt, int N_BODYS, int THREADS_AMOUNT){
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, FR_THETA * dt * 0.5f, N_BODYS);
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, FR_THETA * dt, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, (1 - FR_THETA) * dt * 0.5f, N_BODYS);
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, (1 - 2.0f * FR_THETA) * dt, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, (1 - FR_THETA) * dt * 0.5f, N_BODYS);
    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(coords, accels, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(vels, accels, FR_THETA * dt, N_BODYS);
    gpu_integrateEuler<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(coords, vels, FR_THETA * dt * 0.5f, N_BODYS);
};