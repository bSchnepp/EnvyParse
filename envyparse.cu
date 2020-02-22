/*
Copyright (c) 2019-2020, Brian Schnepp

Permission is hereby granted, free of charge, to any person or organization 
obtaining  a copy of the software and accompanying documentation covered by 
this license (the "Software") to use, reproduce, display, distribute, execute, 
and transmit the Software, and to prepare derivative works of the Software, 
and to permit third-parties to whom the Software is furnished to do so, all 
subject to the following:

The copyright notices in the Software and this entire statement, including 
the above license grant, this restriction and the following disclaimer, must 
be included in all copies of the Software, in whole or in part, and all 
derivative works of the Software, unless such copies or derivative works are 
solely in the form of machine-executable object code generated by a source 
language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT SHALL
THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY 
DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <wchar.h>

#include <cuda.h>
#include <ctype.h>

#include "ngram.h"

/* isspace doesn't compile for device code. 
 * May replace this with a GPU device function later.
 */
#define IS_WHITESPACE(x) (x == ' ' || x == '\n' || x == '\r' || x == '\t' || \
			x == '\v' || x == '\f')


typedef struct GpuContext
{
	/* Team Green calls this a "warp" Insist on calling it a "wavefront". */
	uint64_t GpuIndex;
	uint64_t WavefrontCount;
	uint64_t WavefrontSize; /* Is probably 64??? */
	uint64_t GpuMemoryAmt;
}GpuContext;

/*
 * Copies the entirety of a file to GPU memory.
 *
 * @return void
 *
 * @param Name The name of the file to open.
 * @param Dst A pointer to an unallocated void pointer which
 *	  will be updated with the result of cudaMalloc and
 * 	  the entire content of the file.
 */
void ReadFile(const char *Name, void **Dst, uint64_t *OutSize)
{
	FILE *File = fopen(Name, "r");
	if (File == NULL)
	{
		fprintf(stderr, "Could not open file %s\n", Name);
		*OutSize = 0;
		return;
	}

	fseek(File, 0, SEEK_END);
	uint64_t Length = ftell(File);
	*OutSize = Length;
	rewind(File);

	
	void *Buffer = malloc(Length);
	uint64_t ReadCount = fread(Buffer, 1, Length, File);
	cudaMalloc(Dst, Length);
	cudaMemcpy(*Dst, Buffer, Length, cudaMemcpyHostToDevice);
	free(Buffer);
	fclose(File);
}


__global__ 
void TestFile(void *Src, void *Spaces, uint64_t Length)
{
	uint64_t Index = 0;
	char *SrcC = (char*)(Src);

	for (Index = blockIdx.x * blockDim.y + threadIdx.x;
		Index < Length;
		Index += blockDim.x * gridDim.x)
	{
		if (IS_WHITESPACE(SrcC[Index]))
		{
			((char*)(Spaces))[Index] = 1;
		} else {
			((char*)(Spaces))[Index] = 0;
		}
	}
}


int main(int argc, char **argv)
{
	void *TxtPtr;
	void *SpaceBfr;
	void *TxtPtrHost;

	/* Eat up GPU initalization time early, so sync after doing nothing. */
	cudaDeviceSynchronize();

	if (argc < 2)
	{
		fprintf(stderr, "Error: missing file name to read.\n");
		return -1;
	}
	const char *FName = argv[1]; 

	uint64_t Length;
	ReadFile(FName, &TxtPtr, &Length);
	cudaDeviceSynchronize();

	printf("Got length of file %lu\n", Length);
	cudaMallocManaged(&SpaceBfr, Length);
	TxtPtrHost = malloc(Length);

	/* This performs the best on my GP102. (11GB VRam) */
	cudaMemPrefetchAsync(SpaceBfr, Length, 0);
	TestFile<<<Length/256, 1>>>(TxtPtr, SpaceBfr, Length);
	cudaDeviceSynchronize();
	cudaMemcpy(TxtPtrHost, TxtPtr, Length, cudaMemcpyDeviceToHost);
	for (uint64_t Index = 0; Index < Length; ++Index)
	{
		if (((char*)SpaceBfr)[Index] == 0)
		{
			printf("%c", ((char*)(TxtPtrHost))[Index]);	
		}
	}
	printf("\n");

	cudaFree(TxtPtr);
	cudaFreeHost(SpaceBfr);
	free(TxtPtrHost);
	return 0;
}
