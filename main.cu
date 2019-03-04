#include "src/settings.h"
#include <stdio.h>
#include <time.h>
#include "src/gpuModel.cuh"
#include <unistd.h>
#include <stdbool.h>
extern "C"{
    #include "src/helpers.h"
    #include "src/cpuModel.h"
}

int main(int argc, char* argv[]){
    int startID = 0;
    int N_BODYS = 0;
    int FRAMES_AMOUNT = 0;
    int WRITE_STEP = 0;
    bool benchmark = false;
    bool writeBackups = 0;
    bool useGPU = false;
    char *catName = NULL;
    int THREADS_AMOUNT = 0;

    // supported options
    const char *optString = "s:N:f:w:t:Bbc:Gh?";
    
    int opt = getopt(argc, argv, optString);
    while(opt != -1){
        switch(opt){
            case 's':
                startID = atoi(optarg);
                break;
            case 'N':
                N_BODYS = atoi(optarg);
                break;
            case 'f':
                FRAMES_AMOUNT = atoi(optarg);
                break;
            case 'w':
                WRITE_STEP = atoi(optarg);
                break;
            case 't':
                THREADS_AMOUNT = atoi(optarg);
                break;
            case 'B':
                benchmark = true;
                break;
            case 'b':
                writeBackups = true;
                break;
            case 'c':
                catName = optarg;
                break;
            case 'G':
                useGPU = true;
                break;
            case 'h':
            case '?':
                printHelp();
                return 0;
        };
        opt = getopt(argc, argv, optString);
    };

    if(N_BODYS < 1){
        fprintf(stderr, "ERROR: Too few bodys: %d\n", N_BODYS);
        return -1;
    };
    if(FRAMES_AMOUNT < 1){
        fprintf(stderr, "ERROR: Too few frames: %d\n", FRAMES_AMOUNT);
        return -1;
    };
    if(WRITE_STEP < 1){
        fprintf(stderr, "ERROR: Wrong writing frequency: %d\n", WRITE_STEP);
        return -1;
    };
    if(THREADS_AMOUNT < 1 && useGPU){
        fprintf(stderr, "ERROR: Too small THREADS_AMOUNT: %d\n", THREADS_AMOUNT);
        return -1;
    };


    frame* test = readFrame(catName, N_BODYS);
    int pathLen = sizeof("out/out00000000.csv");
    char path[pathLen];
    
    // for backups
    int backupPathLen = sizeof("backup/back00000000.csv");
    char backupPath[backupPathLen];
    
    if(useGPU){
        test->devBodys = (float4*)cudaProtectedMalloc("devBodys", sizeof(float4) * N_BODYS);
        cudaProtectedMemcpyD("devBodys copy", test->devBodys, test->bodys, sizeof(float4) * N_BODYS);
        
        test->devVels = (float3*)cudaProtectedMalloc("devVels", sizeof(float3) * N_BODYS);
        cudaProtectedMemcpyD("devVels copy", test->devVels, test->vels, sizeof(float3) * N_BODYS);
        
        test->devAccels = (float4*)cudaProtectedMalloc("devAccel", sizeof(float4) * N_BODYS);
    } else {
        test->accels = (float4*)malloc(sizeof(float4) * N_BODYS);
    };
    for(int i = startID; i < startID + FRAMES_AMOUNT; i++){
        for(int j = 0; j < WRITE_STEP; j++){
            if(useGPU){
			    gpu_calculateAccelerations<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT, sizeof(float4) * THREADS_AMOUNT>>>(test->devBodys, test->devAccels, N_BODYS);
			    gpu_updateCoordinates<<<(N_BODYS + THREADS_AMOUNT - 1) / THREADS_AMOUNT, THREADS_AMOUNT>>>(test->devBodys, test->devVels, test->devAccels, DELTA_T);
            } else {
                cpu_calculateAccelerations(test->bodys, test->accels, N_BODYS);
                cpu_updateCoordinates(test->bodys, test->vels, test->accels, DELTA_T, N_BODYS);
            };
        };
        
        if(useGPU){
            cudaProtectedMemcpyH("bodys copy", test->bodys, test->devBodys, sizeof(float4) * N_BODYS);
        };
        
        
        if(sprintf(path, "out/out%08d.csv", i) != pathLen - 1){
            fprintf(stderr, "ERROR: Can't generate filename\n");
            fprintf(stderr, "PathLen: %d\n", pathLen);
            exit(1);
        };
        path[pathLen - 1] = '\0';
        writeFrameShort(path, test, N_BODYS);

        
        if(!benchmark){
            printf("\u001b[1000D");
            int progress = (int)(((float)i / FRAMES_AMOUNT) * 100);
            int i;
            for(i = 0; i < progress; i+=5){
                printf("#");
            };
            for(; i < 100; i+=5){
                printf(" ");
            };

            if(progress % 5 == 0 && writeBackups && !benchmark){
                if(sprintf(backupPath, "backup/back%08d.csv", i) != backupPathLen- 1){
                    fprintf(stderr, "\nERROR: Can't generate filename\n");
                    fprintf(stderr, "PathLen: %d\n", pathLen);
                    exit(1);
                };
                backupPath[backupPathLen - 1] = '\0';
                if(useGPU){
                    cudaProtectedMemcpyH("Backup: vels copy", test->vels, test->devVels, sizeof(float3) *N_BODYS);
                };
                writeFrameFull(backupPath, test, N_BODYS);

            };
        };
        
    };
    
    if(useGPU){
        cudaProtectedMemcpyH("bodys copy", test->bodys, test->devBodys, sizeof(float4) *N_BODYS);
        cudaProtectedMemcpyH("vels copy", test->vels, test->devVels, sizeof(float3) *N_BODYS);
    };
    
    writeFrameFull("result.csv", test, N_BODYS);

    if(!benchmark){
        fprintf(stdout, "DONE\n");
    };

    if(useGPU){
        cudaFree(test->devBodys);
        cudaFree(test->devVels);
    } else {
        free(test->accels);
    };
    freeFrame(test);
    free(test);
    return 0;
};
