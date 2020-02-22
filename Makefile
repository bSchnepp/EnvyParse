NVCC = nvcc

.PHONY: all clean

envyparse: envyparse.cu
	$(NVCC) envyparse.cu -o envyparse

all: envyparse
	

clean:
	rm -rf envyparse
