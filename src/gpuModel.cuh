#ifndef GPU_MODEL_CUH
#define GPU_MODEL_CUH
extern "C"{
	#include "helpers.h"
}

__device__ float3 gpu_calculateBodyBodyInteraction(float4 bodyI, float4 bodyJ, float3 accel);

__device__ float3 gpu_submatrixProcessing(float4 updatingBody, float3 bodyAcceleration);

__global__ void gpu_calculateAccelerations(float4* bodys, float4* accels, int N_BODYS);

__global__ void gpu_updateCoordinatesEuler(float4* bodys, float4* vels, float4* accels, float dt);

__global__ void gpu_integrateEuler(float4 *x, float4 *dx, float dt);

void gpu_updateCoordinatesVelocityVerlet(float4 * coords, float4 * vels, float4 * accels, float dt, int N_BODYS, int THREADS_AMOUNT);

#endif
