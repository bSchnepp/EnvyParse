#include <stdint.h>
#include <stdlib.h>

#include <wchar.h>

#include <cuda.h>

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


int main(int argc, char **argv)
{
	return 0;
}
