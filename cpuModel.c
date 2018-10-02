#include "helpers.h"
#include "cpuModel.h"
#include "settings.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double** prepareGravitationalParameters(double * mass){
    if(mass == NULL){
        fprintf(stderr, "ERROR: mass array is NULL\n");
        exit(1);
    };
    
    double** res = protectedMallocF("Mass Array", sizeof(double*) * N_BODYS);
    for(int i = 0; i < N_BODYS; i++){
        res[i] = protectedMallocF("Mass Array[i]", sizeof(double) * N_BODYS);
        for(int j = 0; j < N_BODYS; j++){
            res[i][j] = (i == j) ? 0 : G * mass[i] * mass[j];
        };
    };

    return res;
};

double** calculateDistArray(double * coord){
    if(coord == NULL){
        fprintf(stderr, "ERROR: Coordinates array is NULL\n");
        exit(1);
    };
    double** res = protectedMallocF("DistArray", sizeof(double*) * N_BODYS);
    for(int i = 0; i < N_BODYS; i++){
        res[i] = protectedMallocF("DistArray[i]", sizeof(double) * N_BODYS);
        for(int j = 0; j < N_BODYS; j++){
            if(fabs(coord[j] - coord[i]) > EPSILON){
                res[i][j] = coord[j] - coord[i];
            } else {
                res[i][j] = EPSILON;
            };
        };
    };
    return res;
};

double** calculateInteraction(const double** masses, double ** dist){
    double** res = protectedMallocF("InterArray", sizeof(double*) * N_BODYS);
    for(int i = 0; i < N_BODYS; i++){
        res[i] = protectedMallocF("InterArray[i]", sizeof(double) * N_BODYS);
        for(int j = 0; j < N_BODYS; j++){
            res[i][j] = masses[i][j] / ((dist[i][j] * dist[i][j]) * (dist[i][j] / fabs(dist[i][j])));
        };
    };
    return res;
};

double* calculateTotalForce(double ** F){
    double * totalForces = protectedMallocF("TotalForces", sizeof(double) * N_BODYS);
    
    // calculate total force
    for(int i = 0; i < N_BODYS; i++){
        totalForces[i] = 0; 
        for(int j = 0; j < N_BODYS; j++){
            totalForces[i] += F[i][j];
        };
    };
    
    return totalForces;
}; 

double* calculateAlteration(double* mass, double * force){
    double * alter = protectedMallocF("Alter", sizeof(double) * N_BODYS);
    for(int i = 0; i < N_BODYS; i++){
        alter[i] = force[i] / mass[i];
    };
    return alter;
};

void integrate(double * x, double * dx){
    for(int i = 0; i < N_BODYS; i++){
        x[i] += DELTA_T * dx[i];
    };
};

void updateCoordinates(double * coord, double * speed, const double ** gravitationalParameters, double * masses){
    // calculating distances
    double ** dist = calculateDistArray(coord);
    
    // calculating interactions
    double ** forces = calculateInteraction(gravitationalParameters, dist);
    
    // calculate total forces
    double * totalF = calculateTotalForce(forces);
    
    // calculate alteration
    double * alt = calculateAlteration(masses, totalF);
    
    // calculate velocity
    integrate(speed, alt);
    
    // calculate new position
    integrate(coord, speed);
    
    // freeing unused allocated memory
    freeSquareMatrix(dist);
    freeSquareMatrix(forces);
    free(totalF);
    free(alt);
};

void updateFrame(frame * fr, const double** gravitationalParameters){
    updateCoordinates(fr->x, fr->vx, gravitationalParameters, fr->masses);
    updateCoordinates(fr->y, fr->vy, gravitationalParameters, fr->masses);
    updateCoordinates(fr->z, fr->vz, gravitationalParameters, fr->masses);
};

void freeSquareMatrix(double ** matrix){
    for(int i = 0; i < N_BODYS; i++){
        free(matrix[i]);
    };
    free(matrix);
};
