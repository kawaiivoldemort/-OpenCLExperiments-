#include <stdio.h>
#include <stdlib.h>

const char* load_file(char* filename) {
	FILE *fp;
	char *source_str;
	size_t source_size, program_size;
	fp = fopen(filename, "rb");
	if (!fp) {
		printf("Failed to load kernel\n");
		return NULL;
	}
	fseek(fp, 0, SEEK_END);
	program_size = ftell(fp);
	rewind(fp);
	source_str = (char*) malloc(program_size + 1);
	source_str[program_size] = '\0';
	fread(source_str, sizeof(char), program_size, fp);
	fclose(fp);
	return source_str;
}