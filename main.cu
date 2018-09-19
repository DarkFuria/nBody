#include "src/settings.h"
#include <stdio.h>
#include <time.h>

extern "C"{
#include "src/helpers.h"
#include "src/cpuModel.h"
}

int main(int argc, char* argv[]){
    if(argc != 4){
        fprintf(stderr, "Usage: ./nbody cat.csv framesAmount writeRate\n");
        exit(1);
    };
    
    frame* test = readFrame(argv[1]);
    int pathLen = sizeof("out/out000000.csv");
    char path[pathLen];
   
    double ** m = prepareGravitationalParameters(test->masses);
    
    int FRAMES_AMOUNT = atoi(argv[2]);
    if(FRAMES_AMOUNT < 1){
        fprintf(stderr, "ERROR: Too small frames amount\n");
        exit(2);
    };
    int WRITE_STEP = atoi(argv[3]);
    if(WRITE_STEP < 1){
        fprintf(stderr, "ERROR: Too small writing step\n");
        exit(1);
    };
    
    unsigned int start = clock();
    
    for(int i = 0; i < FRAMES_AMOUNT; i++){
        for(int j = 0; j < WRITE_STEP; j++){
            updateFrame(test, (const double **)m);
        };
        if(sprintf(path, "out/out%06d.csv", i) != pathLen - 1){
            fprintf(stderr, "ERROR: Can't generate filename\n");
            fprintf(stderr, "PathLen: %d\n", pathLen);
            exit(1);
        };
        path[pathLen - 1] = '\0';
        writeFrameShort(path, test);
        fprintf(stdout, "Frame#%05d created\n", i);
    };
    unsigned int stop = clock();
    double avg = (double)(stop - start)/ CLOCKS_PER_SEC;
    avg/=FRAMES_AMOUNT;
    fprintf(stdout, "Average time per frame: %f s\n", avg);
    
    writeFrameFull("result.csv", test);
   
    freeSquareMatrix(m);
    freeFrame(test);
    free(test);
    return 0;
};
