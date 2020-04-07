CC=nvcc
CFLAGS=-arch=sm_35 -rdc=true --expt-relaxed-constexpr

build: main.cu 
	$(CC) main.cu $(CFLAGS) $(OPTIM) $(DEBUG) $(ERROR)

release: OPTIM=-O2

release: build

debug: DEBUG=-g -lineinfo

debug: build

error: ERROR=--compiler-options -Wall

error: build
