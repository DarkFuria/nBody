CC=nvcc
SRC=main.cu src/helpers.o src/cpuModel.o src/gpuModel.o
FLAGS=-lm -arch=sm_61 --restrict

all: src/helpers.o src/cpuModel.o src/gpuModel.o
	$(CC) $(SRC) -o nbody -g $(FLAGS)

src/helpers.o: src/helpers.c
	$(CC) -c src/helpers.c
	mv helpers.o src/helpers.o
	
src/cpuModel.o: src/cpuModel.c
	$(CC) -c src/cpuModel.c $(FLAGS)
	mv cpuModel.o src/cpuModel.o
	
src/gpuModel.o: src/gpuModel.cu
	$(CC) -c src/gpuModel.cu $(FLAGS)
	mv gpuModel.o src/gpuModel.o
	
check:
	valgrind --track-origins=yes --leak-check=full ./nbody catalogue1024.csv 1 1

clean:
	rm nbody
