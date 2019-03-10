CC=nvcc
SRC=main.cu src/helpers.o src/gpuModel.o src/cpuModel.o
FLAGS=-lm -arch=sm_61 -O0
CPU_FLAGS=-O0 -lm
CPU_CC=gcc-6


all: src/helpers.o src/gpuModel.o src/cpuModel.o
	$(CC) $(SRC) -o nbody $(FLAGS)
	rm src/*.o

src/helpers.o: src/helpers.c
	$(CC) -c src/helpers.c
	mv helpers.o src/helpers.o
	
	
src/gpuModel.o: src/gpuModel.cu
	$(CC) -c src/gpuModel.cu $(FLAGS)
	mv gpuModel.o src/gpuModel.o

src/cpuModel.o: src/cpuModel.c
	$(CPU_CC) -c src/cpuModel.c $(CPU_FLAGS)
	mv cpuModel.o src/cpuModel.o

clean:
	rm nbody
	rm src/*.o
