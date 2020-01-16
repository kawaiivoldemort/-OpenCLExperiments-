#include <CL/cl.h>

typedef struct oCL_data {
	cl_command_queue queue;
	cl_context context;
	cl_program program;
} oCL_data;

void verify_result(const int, const int, double*, double*);
void lu_decomposition(const int, const int, double*);
void lu_decomposition2(const int, const int, double*);
void lu_decomposition_ocl(const int, const int, double*, oCL_data);
void lu_decomposition_ocl2(const int, const int, double*, oCL_data);