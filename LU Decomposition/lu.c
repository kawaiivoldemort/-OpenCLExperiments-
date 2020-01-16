#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lu.h"
#include "load_file.h"

void lu_decomposition(
    /*
     * LU decomposition without pivoting
     * In-place decomposition of form A=LU
     * L is returned below main diagonal of A
     * U is returned at and above main diagonal
     */
    const int n,
    const int lda,
    double* A
) {
    for(int i = 1; i < n; i++) {
        double* Ai = (A + i*lda);
        for(int k = 0; k < i; k++) {
            double* Ak = (A + k*lda);
            double Aik = Ai[k] / Ak[k];
            for(int j = k + 1; j < n; j++) {
                Ai[j] -= Aik * Ak[j];
            }
            Ai[k] = Aik;
        }
    }
}

void lu_decomposition2(
    /*
     * LU decomposition without pivoting
     * In-place decomposition of form A=LU
     * L is returned below main diagonal of A
     * U is returned at and above main diagonal
     */
    const int n,
    const int lda,
    double* A
) {
    for(int k = 0; k < n; k++) {
        for(int i = k + 1; i < n; i++) {
            A[i*lda + k] /= A[k*lda + k];
        }
        for(int i = k + 1; i < n; i++) {
            double Aik = A[i*lda + k];
            for(int j = k + 1; j < n; j++) {
                A[i*lda + j] -= Aik * A[k*lda + j];
            }
        }
    }
}

void lu_decomposition_ocl(
    const int n,
    const int lda,
    double* h_A,
    oCL_data data
) {
	cl_int err;
	// Allocate buffers for read and write and copy
	cl_mem d_A = clCreateBuffer(
		data.context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(double) * lda * n,
		h_A,
		NULL
	);
    double* d_Ares = calloc(sizeof(double), lda*n);
    int np = 4;
	// Create the kernel and set arguments
	cl_kernel kernel = clCreateKernel(data.program, "set_aik", &err);
	cl_kernel kernel2 = clCreateKernel(data.program, "lu_decomposition_ocl", &err);
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    err  |= clSetKernelArg(kernel, 3, sizeof(int), &lda);
	err  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &d_A);
    err  |= clSetKernelArg(kernel2, 3, sizeof(int), &lda);
    for(int i = 1; i < n; i++) {
	    err  |= clSetKernelArg(kernel, 1, sizeof(int), &i);
	    err  |= clSetKernelArg(kernel2, 1, sizeof(int), &i);
        for(int k = 0; k < i; k++) {
            int count = n - k - 1;
	        err  |= clSetKernelArg(kernel, 2, sizeof(int), &k);
	        err  |= clSetKernelArg(kernel2, 2, sizeof(int), &k);
	        err  |= clSetKernelArg(kernel2, 4, sizeof(int), &count);
	        err  |= clSetKernelArg(kernel2, 5, sizeof(int), &np);
            // Set the work size	
            size_t global_work_size1[] = { 1 };
            // Enqueue the kernel as a command
            err = clEnqueueNDRangeKernel(
                data.queue,
                kernel,
                1,
                NULL,
                global_work_size1,
                NULL,
                0,
                NULL,
                NULL
            );
            // Complete the command
            clFinish(data.queue);
            size_t global_work_size[] = { np };
            // Enqueue the kernel as a command
            err = clEnqueueNDRangeKernel(
                data.queue,
                kernel2,
                1,
                NULL,
                global_work_size,
                NULL,
                0,
                NULL,
                NULL
            );
            // Complete the command
            clFinish(data.queue);
        }
    }
	// Copy back
	clEnqueueReadBuffer(
		data.queue,
		d_A,
		CL_TRUE,
		0,
		sizeof(double) * lda * n,
		h_A,
		0,
		NULL,
		NULL
	);
	// Clean up
	clReleaseMemObject(d_A);
	clReleaseKernel(kernel);
	clReleaseKernel(kernel2);
}

void lu_decomposition_ocl2(
    const int n,
    const int lda,
    double* h_A,
    oCL_data data
) {
	cl_int err;
	// Allocate buffers for read and write and copy
	cl_mem d_A = clCreateBuffer(
		data.context,
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
		sizeof(double) * lda * n,
		h_A,
		NULL
	);
    double* d_Ares = calloc(sizeof(double), lda*n);
    int np = 4;
	// Create the kernel and set arguments
	cl_kernel kernel = clCreateKernel(data.program, "lu_decomposition_ocl2", &err);
	cl_kernel kernel2 = clCreateKernel(data.program, "set_aik2", &err);
	err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    err  |= clSetKernelArg(kernel, 2, sizeof(int), &lda);
    err  |= clSetKernelArg(kernel, 3, sizeof(int), &n);
	err  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &d_A);
    err  |= clSetKernelArg(kernel2, 2, sizeof(int), &lda);
    err  |= clSetKernelArg(kernel2, 3, sizeof(int), &n);
    for(int k = 0; k < n; k++) {
        size_t global_work_size2[] = { 1 };
        // Enqueue the kernel as a command
        err  |= clSetKernelArg(kernel2, 1, sizeof(int), &k);
        err = clEnqueueNDRangeKernel(
            data.queue,
            kernel2,
            1,
            NULL,
            global_work_size2,
            NULL,
            0,
            NULL,
            NULL
        );
        // Complete the command
        clFinish(data.queue);
        // Enqueue the kernel as a command
        err  |= clSetKernelArg(kernel, 1, sizeof(int), &k);
        size_t global_work_size[] = { n - k - 1 };
        err = clEnqueueNDRangeKernel(
            data.queue,
            kernel,
            1,
            NULL,
            global_work_size,
            NULL,
            0,
            NULL,
            NULL
        );
        // Complete the command
        clFinish(data.queue);
    }
	// Copy back
	clEnqueueReadBuffer(
		data.queue,
		d_A,
		CL_TRUE,
		0,
		sizeof(double) * lda * n,
		h_A,
		0,
		NULL,
		NULL
	);
	// Clean up
	clReleaseMemObject(d_A);
	clReleaseKernel(kernel);
}

void verify_result(
    const int n,
    const int lda,
    double* LU,
    double* refA
) {
    // Verifying that A = LU
    double* A = (double*) calloc(sizeof(double)*n*lda, 64);
    double* L = (double*) calloc(sizeof(double)*n*lda, 64);
    double* U = (double*) calloc(sizeof(double)*n*lda, 64);
    for(int i = 0; i < lda; i++) {
        A[i] = 0.0f;
        L[i] = 0.0f;
        U[i] = 0.0f;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            L[i*lda + j] = LU[i*lda + j];
        }
        L[i*lda+i] = 1.0f;
        for (int j = i; j < n; j++) {
            U[i*lda + j] = LU[i*lda + j];
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                A[i*lda + j] += L[i*lda + k]*U[k*lda + j];
            }
        }
    }

    double deviation1 = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            deviation1 += (refA[i*lda+j] - A[i*lda+j])*(refA[i*lda+j] - A[i*lda+j]);
        }
    }
    deviation1 /= (double)(n*lda);
    if (isnan(deviation1) || (deviation1 > 1.0e-2)) {
        printf("ERROR: LU is not equal to A (deviation1=%e)!\n", deviation1);
        exit(1);
    }

#ifdef VERBOSE
    printf("\n(L-D)+U:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.3e", LU[i*lda+j]);
        }
        printf("\n");
    }

    printf("\nL:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.3e", L[i*lda+j]);
        }
        printf("\n");
    }

    printf("\nU:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.3e", U[i*lda+j]);
        }
        printf("\n");
    }

    printf("\nLU:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.3e", A[i*lda+j]);
        }
        printf("\n");
    }

    printf("\nA:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.3e", refA[i*lda+j]);
        }
        printf("\n");
    }

    printf("deviation1=%e\n", deviation1);
#endif

    free(A);
    free(L);
    free(U);
}