#include "helpers.h"
#include "cpuModel.h"
#include "settings.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

float4 cpu_calculateInteraction(float4 bodyI, float4 bodyJ, float4 accel){
    // calculating distances [11 FLOPS]
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

void cpu_calculateAccelerations(float4 * bodys, float4 *accels, int N_BODYS){
    for(int i = 0; i < N_BODYS; i++){
        float4 accel = {0.0f, 0.0f, 0.0f, 0.0f};
        for(int j = 0; j < N_BODYS; j++){
            cpu_calculateInteraction(bodys[i], bodys[j], accel);
        };
        accels[i] = accel;
    };
};

void cpu_updateCoordinatesEuler(float4 * coords, float4 * vels, float4 * accels, float dt, int N_BODYS){
    for(int i = 0; i < N_BODYS; i++){
        // updating veloсity
        vels[i].x += accels[i].x * dt;
        vels[i].y += accels[i].y * dt;
        vels[i].z += accels[i].z * dt;

        // updating coords
        coords[i].x += vels[i].x * dt;
        coords[i].y += vels[i].y * dt;
        coords[i].z += vels[i].z * dt;
    };  
};

void cpu_integrateEuler(float4 *x, float4 *dx, float dt, int N_BODYS){
    for(int i = 0; i < N_BODYS; i++){
        x[i].x += dx[i].x * dt;
        x[i].y += dx[i].y * dt;
        x[i].z += dx[i].z * dt;
    };
};

void cpu_updateCoordinatesVelocityVerlet(float4 * coords, float4 * vels, float4 * accels, float dt, int N_BODYS){
    cpu_calculateAccelerations(coords, accels, N_BODYS);
    cpu_integrateEuler(vels, accels, dt / 2.0, N_BODYS);
    cpu_integrateEuler(coords, vels, dt, N_BODYS);
    cpu_calculateAccelerations(coords, accels, N_BODYS);
    cpu_integrateEuler(vels, accels, dt / 2.0, N_BODYS);
};