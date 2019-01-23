#ifndef GPU_MODEL_CUH
#define GPU_MODEL_CUH
extern "C"{
	#include "helpers.h"
}

__device__ float3 calculateBodyBodyInteraction(float4 bodyI, float4 bodyJ, float3 accel);

__device__ float3 submatrixProcessing(float4 updatingBody, float3 bodyAcceleration);

__global__ void calculateAccelerations(float4* bodys, float4* accels, int N_BODYS);

__global__ void updateCoordinates(float4* bodys, float3* vels, float4* accels, float dt);

#endif
