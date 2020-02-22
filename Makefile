NVCC = nvcc
CFLAGS = -Iinc

.PHONY: all clean

envyparse: envyparse.cu
	$(NVCC) $(CFLAGS) envyparse.cu -o envyparse

all: envyparse
	

clean:
	rm -rf envyparse
