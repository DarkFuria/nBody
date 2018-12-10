#include "src/settings.h"
#include <stdio.h>
#include <time.h>
#include "src/gpuModel.cuh"
extern "C"{
    #include "src/helpers.h"
}

int main(int argc, char* argv[]){
    int startID = 0;
    int writeBackups = 1;
    if(argc < 3){
        fprintf(stderr, "Usage: ./nbody cat.csv framesAmount writeRate [writeBackups = 1] [startID]\n");
        exit(1);
    } else if (argc == 5){
        writeBackups = atoi(argv[4]);
    } else if (argc == 6){
        writeBackups = atoi(argv[4]);
        startID = atoi(argv[5]);
    };
    
    frame* test = readFrame(argv[1]);
    int pathLen = sizeof("out/out00000000.csv");
    char path[pathLen];
    
    // for backups
    int backupPathLen = sizeof("backup/back00000000.csv");
    char backupPath[backupPathLen];
    
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
    
    test->devBodys = (float4*)cudaProtectedMalloc("devBodys", sizeof(float4) * N_BODYS);
    cudaProtectedMemcpyD("devBodys copy", test->devBodys, test->bodys, sizeof(float4) * N_BODYS);
    
    test->devVels = (float3*)cudaProtectedMalloc("devVels", sizeof(float3) * N_BODYS);
    cudaProtectedMemcpyD("devVels copy", test->devVels, test->vels, sizeof(float3) * N_BODYS);
    
    test->devAccels = (float4*)cudaProtectedMalloc("devAccel", sizeof(float4) * N_BODYS);
    
    for(int i = startID; i < startID + FRAMES_AMOUNT; i++){
        for(int j = 0; j < WRITE_STEP; j++){
				calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(test->devBodys, test->devAccels);
				updateCoordinates<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(test->devBodys, test->devVels, test->devAccels, DELTA_T);
        };
        
        cudaProtectedMemcpyH("bodys copy", test->bodys, test->devBodys, sizeof(float4) *N_BODYS);
        
        
        if(sprintf(path, "out/out%08d.csv", i) != pathLen - 1){
            fprintf(stderr, "ERROR: Can't generate filename\n");
            fprintf(stderr, "PathLen: %d\n", pathLen);
            exit(1);
        };
        path[pathLen - 1] = '\0';
        writeFrameShort(path, test);

        if(i % (FRAMES_AMOUNT / 100) == 0){
            fprintf(stdout, "%d frames out of %d done\n", i+1 - startID, FRAMES_AMOUNT);
        };

        if(i % (FRAMES_AMOUNT / 100) == 0 && writeBackups){
            if(sprintf(backupPath, "backup/back%08d.csv", i) != backupPathLen- 1){
                fprintf(stderr, "ERROR: Can't generate filename\n");
                fprintf(stderr, "PathLen: %d\n", pathLen);
                exit(1);
            };
            backupPath[backupPathLen - 1] = '\0';
            cudaProtectedMemcpyH("Backup: vels copy", test->vels, test->devVels, sizeof(float3) *N_BODYS);
            writeFrameFull(backupPath, test);

        };
    };
    
    cudaProtectedMemcpyH("vels copy", test->vels, test->devVels, sizeof(float3) *N_BODYS);
    
    writeFrameFull("result.csv", test);

    fprintf(stdout, "DONE\n");

    cudaFree(test->devBodys);
    cudaFree(test->devVels);
    freeFrame(test);
    free(test);
    return 0;
};
