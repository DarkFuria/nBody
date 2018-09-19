#ifndef CPU_MODEL_H
#define CPU_MODEL_H

// creates matrix, each element is G * mass[i] * mass[j], there i and j - indexes of interacting bodys
// if i == j, matrix[i][j] == 0 
double** prepareGravitationalParameters(double * mass);

// calculates distance between elements in projection onto X, Y or Z axis
double** calculateDistArray(double * coord);

// calculates interactions projections onto axis based on gravitation law
// F = G * M1 * M2 / R^2
double** calculateInteraction(const double** masses, double ** dist);

// Calculates vector sum of forces acting on each body
double* calculateTotalForce(double ** F);

// Calculates alteration of the body
// a[i] = force[i] / mass[i]
double* calculateAlteration(double* mass, double * force);

// Integrates x using numerical methods
void integrate(double * x, double * dx);

// recalclulates coordinates
void updateCoordinates(double * coord, double * speed, const double ** gravitationalParameters, double * masses);

// updates frame
void updateFrame(frame * fr, const double** masses);

// frees created matrix
void freeSquareMatrix(double ** matrix);

#endif
