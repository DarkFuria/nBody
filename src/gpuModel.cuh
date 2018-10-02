#ifndef GPU_MODEL_CUH
#define GPU_MODEL_CUH
extern "C"{
#include "helpers.h"
}

typedef struct {
	double* dist;
	double* interactions;
	double* TF;
	double* alters;
} tempData;

// creates temporary data structure 
tempData* createTempData();

// frees temporary data structure
void freeTempData(tempData* td);

// takes masses array and returns it into predefined array(N*N)
__global__ void gpu_prepareGravitationalParameters(double* __restrict__ gravitationalParameters, const double* __restrict__ masses);

// calculates distance between all points int projection on the one axis
__global__ void gpu_calculateDistArray(double* __restrict__ dist, double* __restrict__ coordinates);

// calculates interactions using Gravitation Law
__global__ void gpu_calculateInteraction(double * __restrict__ forces, const double * __restrict__ gravitationalParameters, const double * __restrict__ dist);

// calcilate vector sum of forces in projection on the axis
__global__ void gpu_calculateTotalForces(double * __restrict__ totalForces, double * __restrict__ forces);

// calculates alterations
__global__ void gpu_calculateAlteration(double * alteration, double * __restrict__ totalForces, double * __restrict__ masses);

// integrates using Eulers method
__global__ void gpu_integrate(double * x, double * __restrict__ dx);

// updates coordinates
void gpu_updateCoordinates(double * coord, double * vels, const double * __restrict__ gravitationalParameters, double * __restrict__ masses, const tempData* __restrict__ td);

// updates frame
void gpu_updateFrame(frame* fr, double * __restrict__ gravitationalParameters, const tempData* __restrict__ td);
#endif
