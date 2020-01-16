#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>
#include "lu.h"
#include "load_file.h"
#include <CL/cl.h>

void print_m(double* A, int lda, int n) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			printf("%d ", *(A + i*lda + j));
		}
		printf("\n");
	}
}

double* copy(double* A, int lda, int n) {
	double* A2 = calloc(n*lda, sizeof(double));
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			*(A2 + i*lda + j) = *(A + i*lda + j);
		}
	}
	return A2;
}

int main(const int argc, const char** argv) {
    // Error value
	cl_int err;
	// Get the platforms found based on the number of platforms to work on
	cl_platform_id platform;
	cl_uint num_platforms;
	err = clGetPlatformIDs(
		1,
		&platform,
		&num_platforms
	);
	// Get the devices on the platform
	cl_device_id device;
	err = clGetDeviceIDs(
		platform,
		CL_DEVICE_TYPE_CPU,
		1,
		&device,
		NULL
	);
	// Create the context to execute in
	cl_context_properties properties[] = {
		(cl_context_properties) CL_CONTEXT_PLATFORM,
		(cl_context_properties) platform,
		(cl_context_properties) 0
	};
	cl_context context = clCreateContext(
		properties,
		1,
		&device,
		NULL,
		NULL,
		&err
	);
	// Create a command queue to execute on the device
	cl_command_queue queue = clCreateCommandQueueWithProperties(
		context,
		device,
		0,
		&err
	);
	// Create the program
	const char* kernel_source = load_file("kernel.cl");
	cl_program program = clCreateProgramWithSource(
		context,
		1,
		&kernel_source,
		NULL,
		&err
	);
	// Build the program	
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS) {
		// error
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(
			program,
			device,
			CL_PROGRAM_BUILD_LOG,
			sizeof(buffer),
			buffer,
			&len
		);
		printf(buffer);
	}

	oCL_data data = {
		queue,
		context,
		program
	};

	// Problem size and other parameters
	const int n=512;
	const int lda=528;
	const int nMatrices=100;
	const double HztoPerf = 1e-9*2.0/3.0*((double)n*n*lda)*nMatrices;

	const size_t containerSize = sizeof(double)*n*lda+64;
	double* A = (double*) calloc(containerSize, 64);
	double* referenceMatrix = (double*) calloc(containerSize, 64);

  // Initialize matrix
	for (int i = 0; i < n; i++) {
		double sum = 0.0f;
		for (int j = 0; j < n; j++) {
			A[i*lda+j] = (double)(i*n+j);
			sum += A[i*lda+j];
		}
		sum -= A[i*lda+i];
		A[i*lda+i] = 2.0f*sum;
	}
	A[(n-1)*lda+n] = 0.0f; // Touch just in case
	for(int i = 0; i < n*lda; i++) {
		referenceMatrix[i] = A[i];
	}

	// Perform benchmark
	printf("LU decomposition of %d matrices of size %dx%d on %s...\n\n", 
		nMatrices, n, n,
#ifndef __MIC__
		"CPU"
#else
		"MIC"
#endif
	 );

	double rate = 0, dRate = 0; // Benchmarking data
	const int nTrials = 10;
	const int skipTrials = 3; // First step is warm-up on Xeon Phi coprocessor
	printf("\033[1m%5s %10s %8s\033[0m\n", "Trial", "Time, s", "GFLOP/s");

	// Verify result
	lu_decomposition2(n, lda, A);
	verify_result(n, lda, A, referenceMatrix);

	// Measure performance
	for (int trial = 1; trial <= nTrials; trial++) {

		const double tStart = omp_get_wtime(); // Start timing
		// Benchmarking multiple decompositions to improve statistics
		for (int m = 0; m < nMatrices; m++) {
			lu_decomposition2(n, lda, A);
		}
		const double tEnd = omp_get_wtime(); // End timing

		if (trial > skipTrials) { // Collect statistxics
			rate  += HztoPerf/(tEnd - tStart); 
			dRate += HztoPerf*HztoPerf/((tEnd - tStart)*(tEnd-tStart)); 
		}

		printf("%5d %10.3e %8.2f %s\n", 
			trial, (tEnd-tStart), HztoPerf/(tEnd-tStart), (trial<=skipTrials?"*":""));
		fflush(stdout);
	}
	rate /= (double)(nTrials-skipTrials); 
	dRate = sqrt(dRate/(double)(nTrials-skipTrials)-rate*rate);
	printf("-----------------------------------------------------\n");
	printf("\033[1m%s %4s \033[42m%10.2f +- %.2f GFLOP/s\033[0m\n",
		"Average performance:", "", rate, dRate);
	printf("-----------------------------------------------------\n");
	printf("* - warm-up, not included in average\n\n");

	free(A);
	free(referenceMatrix);

	
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}