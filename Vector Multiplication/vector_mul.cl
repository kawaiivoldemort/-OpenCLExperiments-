__kernel
void vector_mul(
    __global const int* a,
    __global const int* b,
    __global       int* c,
             const int  j,
             const int  k
) {
    int m = get_global_id(0);
    for(int o = 0; o < k; o++) {
        int result = 0;
        for(int n = 0; n < j; n++) {
            result += a[m * j + n] * b[n * k + o];
        }
        c[m * k + o] = result;
    }
}