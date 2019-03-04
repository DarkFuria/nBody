#ifndef CPU_MODEL_H
#define CPU_MODEL_H

float4 cpu_calculateInteraction(float4 bodyI, float4 bodyJ, float4 accel);

void cpu_calculateAccelerations(float4 * bodys, float4 *accels, int N_BODYS);

void cpu_updateCoordinates(float4 * bodys, float3 * vels, float4 * accels, float dt, int N_BODYS);

#endif
