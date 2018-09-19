#include "helpers.h"

#include <stdlib.h>
#include <stdio.h>
#include "settings.h"

void * protectedMallocF(char const* arrName, unsigned int size){
    float * tmp = malloc(size);
    if(tmp == NULL){
        fprintf(stderr, "ERROR: Couldnt allocate memory for %s\n", arrName);
        exit(1);
    };
    return tmp;
};

frame * readFrame(char const* frameName){
    FILE *inp;
    inp = fopen(frameName, "r");
    if(inp == NULL){
        fprintf(stderr, "ERROR: Can't open file %s\n", frameName);
        exit(1);
    };
    
    frame *tmp = malloc(sizeof(frame));
    if(tmp == NULL){
        fprintf(stderr, "ERROR: Couldnt allocate memory for new frame\n");
        fclose(inp);
        exit(1);
    };
    
    tmp->masses = protectedMallocF("tmp->masses", sizeof(double) * N_BODYS);
    tmp->x = protectedMallocF("tmp->x", sizeof(double) * N_BODYS);
    tmp->y = protectedMallocF("tmp->y", sizeof(double) * N_BODYS);
    tmp->z = protectedMallocF("tmp->z", sizeof(double) * N_BODYS);
    tmp->vx = protectedMallocF("tmp->vx", sizeof(double) * N_BODYS);
    tmp->vy = protectedMallocF("tmp->vy", sizeof(double) * N_BODYS);
    tmp->vz = protectedMallocF("tmp->vz", sizeof(double) * N_BODYS);
                    
                    
    for(int i = 0; i < N_BODYS; i++){
        if(fscanf(inp, "%lf %lf %lf %lf %lf %lf %lf\n",&tmp->masses[i], &tmp->x[i], &tmp->y[i], &tmp->z[i], &tmp->vx[i], &tmp->vy[i], &tmp->vz[i]) != 7) {
            fprintf(stderr, "ERROR: Can't read file %s\n", frameName);
            free(fr->masses);
            free(fr->x);
            free(fr->y);
            free(fr->z);
            free(fr->vx);
            free(fr->vy);
            free(fr->vz);
            fclose(inp);
            exit(1);
        };
    };
    
    fclose(inp);
    return tmp;
};

void printFrame(frame const* fr){
    for(int i = 0; i < N_BODYS; i++){
        fprintf(stdout, "%f %f %f %f %f %f\n", fr->x[i], fr->y[i], fr->z[i], fr->vx[i], fr->vy[i], fr->vz[i]);
    };
};

void printSquareMatrix(const double ** matrix){
    for(int i = 0; i < N_BODYS; i++){
        for(int j = 0; j < N_BODYS; j++){
            fprintf(stdout, "%15.2f", matrix[i][j]);
        };
        fprintf(stdout, "\n");
    };
};

void writeFrameFull(char const* frameName,const frame* fr ){
    FILE * out = fopen(frameName, "w");
    if(out == NULL){
        fprintf(stderr, "ERROR: Can't open file %s\n", frameName);
        exit(1);
    };
    for(int i = 0; i < N_BODYS; i++){
        fprintf(out, "%f %f %f %f %f %f %f\n", fr->masses[i], fr->x[i], fr->y[i], fr->z[i], fr->vx[i], fr->vy[i], fr->vz[i]);
    };
    fclose(out);
};

void writeFrameShort(char const* frameName,const frame* fr ){
    FILE * out = fopen(frameName, "w");
    if(out == NULL){
        fprintf(stderr, "ERROR: Can't open file %s\n", frameName);
        exit(1);
    };
    for(int i = 0; i < N_BODYS; i++){
        fprintf(out, "%f %f %f\n", fr->x[i], fr->y[i], fr->z[i]);
    };
    fclose(out);
};

void freeFrame(frame* fr){
    free(fr->masses);
    free(fr->x);
    free(fr->y);
    free(fr->z);
    free(fr->vx);
    free(fr->vy);
    free(fr->vz);
};
