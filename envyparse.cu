#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <wchar.h>

#include <cuda.h>
#include <ctype.h>

typedef struct Ngram
{
	uint16_t Length;
	const wchar_t *Content;
}Ngram;

typedef struct Word
{
	uint16_t Length;
	Ngram *Content;
}Word;

/* isspace doesn't compile for device code. */
#define IS_WHITESPACE(x) (x == ' ' || x == '\n' || x == '\r' || x == '\t' || \
			x == '\v' || x == '\f')

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

__global__ void TestFile(void *Src, void *Spaces, uint64_t Length)
{
	uint64_t Index = blockIdx.x * blockDim.y + threadIdx.x;
	char *SrcC = (char*)(Src);

	if (Index < Length)
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

	uint64_t Length;
	ReadFile("data_test.txt", &TxtPtr, &Length);
	cudaDeviceSynchronize();

	printf("Got length of file %lu\n", Length);
	cudaMallocHost(&SpaceBfr, Length);
	TxtPtrHost = malloc(Length);

	TestFile<<<1, Length>>>(TxtPtr, SpaceBfr, Length);
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
