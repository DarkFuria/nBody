// model parameters
#define DELTA_T 86400 // seconds

// CONSTANTS
#define G 6.67e-11
#define EPSILON 1e6 // helps to avoid singularity


// Forest-Ruth algorithm constant
#define FR_THETA 1.3512071919


enum INTEGRATOR_TYPE {
    EULER = 1,
    VELOCITYVERLET = 2,
    FORESTRUTH = 3
};