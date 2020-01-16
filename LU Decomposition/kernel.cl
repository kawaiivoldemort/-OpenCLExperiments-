__kernel void set_aik(
    __global double* A,
    const int i,
    const int k,
    const int lda
) {
    A[i*lda + k] = A[i*lda + k] / A[k*lda + k];
}

__kernel void lu_decomposition_ocl(
    __global double* A,
    const int i,
    const int k,
    const int lda,
    const int count,
    const int np
) {
    __global double* Ai = &A[i * lda];
    __global double* Ak = &A[k * lda];
    int gid = get_global_id(0);
    int j = gid * (count / np);
    j += k + 1;
    int u = j + count / np;
    if(gid == (np - 1)) {
        u += count % np;
    }
    double Aik = Ai[k];
    for(; j < u; j++) {
        Ai[j] -= Aik * Ak[j];
    }
}

__kernel void set_aik2(
    __global double* A,
    const int k,
    const int lda,
    const int n
) {
    for(int i = k + 1; i < n; i++) {
        A[i*lda + k] /= A[k*lda + k];
    }
}

__kernel void lu_decomposition_ocl2(
    __global double* A,
    const int k,
    const int lda,
    const int n
) {
    int i = get_global_id(0) + k + 1;
    __global double* Ai = (A + i*lda);
    __global double* Ak = (A + k*lda);    
    double Aik = Ai[k];
    for(int j = k + 1; j < n; j++) {
        Ai[j] -= Aik * Ak[j];
    }
}