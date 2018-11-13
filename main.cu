#include "src/settings.h"
#include <stdio.h>
#include <time.h>
#include "src/gpuModel.cuh"
extern "C"{
    #include "src/helpers.h"
}

int main(int argc, char* argv[]){
    if(argc != 4){
        fprintf(stderr, "Usage: ./nbody cat.csv framesAmount writeRate\n");
        exit(1);
    };
    
    frame* test = readFrame(argv[1]);
    int pathLen = sizeof("out/out00000000.csv");
    char path[pathLen];
    double * gravitationalParameters;
    
    
    int FRAMES_AMOUNT = atoi(argv[2]);
    if(FRAMES_AMOUNT < 1){
        fprintf(stderr, "ERROR: Too small frames amount\n");
        exit(1);
    };
    int WRITE_STEP = atoi(argv[3]);
    if(WRITE_STEP < 1){
        fprintf(stderr, "ERROR: Too small writing step\n");
        exit(1);
    };
    
    test->devMasses = cudaProtectedMalloc("massesGP", sizeof(double) * N_BODYS);
    cudaProtectedMemcpyD("massesGP", test->devMasses, test->masses, sizeof(double) * N_BODYS);
    
    test->devX = cudaProtectedMalloc("devX", sizeof(double) * N_BODYS);
    cudaProtectedMemcpyD("devX", test->devX, test->x, sizeof(double) * N_BODYS);
    
    test->devY = cudaProtectedMalloc("devY", sizeof(double) * N_BODYS);
    cudaProtectedMemcpyD("devY", test->devY, test->y, sizeof(double) * N_BODYS);
    
    test->devZ = cudaProtectedMalloc("devZ", sizeof(double) * N_BODYS);
    cudaProtectedMemcpyD("devZ", test->devZ, test->z, sizeof(double) * N_BODYS);
    
    test->devVx = cudaProtectedMalloc("devVx", sizeof(double) * N_BODYS);
    cudaProtectedMemcpyD("devVx", test->devVx, test->vx, sizeof(double) * N_BODYS);
    
    test->devVy = cudaProtectedMalloc("devVy", sizeof(double) * N_BODYS);
    cudaProtectedMemcpyD("devVy", test->devVy, test->vy, sizeof(double) * N_BODYS);
    
    test->devVz = cudaProtectedMalloc("devVz", sizeof(double) * N_BODYS);
    cudaProtectedMemcpyD("devVz", test->devVz, test->vz, sizeof(double) * N_BODYS);
    
    gravitationalParameters = cudaProtectedMalloc("gravitationalParameters", sizeof(double) * N_BODYS * N_BODYS);
    gpu_prepareGravitationalParameters<<<(N_BODYS * N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(gravitationalParameters, test->devMasses);
    cudaDeviceSynchronize();
    
    tempData* td = createTempData();
    
    for(int i = 0; i < FRAMES_AMOUNT; i++){
        for(int j = 0; j < WRITE_STEP; j++){
			for(int k = 0; k < 1000; k++){
				gpu_updateFrame(test, gravitationalParameters, td);
			};
        };
        
        cudaProtectedMemcpyH("X copy", test->x, test->devX, sizeof(double) *N_BODYS);
        cudaProtectedMemcpyH("Y copy", test->y, test->devY, sizeof(double) *N_BODYS);
        cudaProtectedMemcpyH("Z copy", test->z, test->devZ, sizeof(double) *N_BODYS);
        
        
        if(sprintf(path, "out/out%08d.csv", i) != pathLen - 1){
            fprintf(stderr, "ERROR: Can't generate filename\n");
            fprintf(stderr, "PathLen: %d\n", pathLen);
            exit(1);
        };
        path[pathLen - 1] = '\0';
        writeFrameShort(path, test);
        fprintf(stdout, "Frame#%08d created\n", i);
    };
    
    cudaProtectedMemcpyH("vX copy", test->vx, test->devVx, sizeof(double) *N_BODYS);
    cudaProtectedMemcpyH("vY copy", test->vy, test->devVy, sizeof(double) *N_BODYS);
    cudaProtectedMemcpyH("vZ copy", test->vz, test->devVz, sizeof(double) *N_BODYS);
    
    writeFrameFull("result.csv", test);
   
	freeTempData(td);
	free(td);
    cudaFree(gravitationalParameters);
    cudaFree(test->devX);
    cudaFree(test->devY);
    cudaFree(test->devZ);
    cudaFree(test->devVx);
    cudaFree(test->devVy);
    cudaFree(test->devVz);
    cudaFree(test->devMasses);
    freeFrame(test);
    free(test);
    return 0;
};
