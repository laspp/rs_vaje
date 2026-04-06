#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <riscv_vector.h>
// Include the gem5 m5ops header file
#include <gem5/m5ops.h>


float dot_scalar(const float *x, const float *y, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++)
        sum += x[i] * y[i];
    return sum;
}


float dot_rvv_naive(const float *x, const float *y, size_t n) {
    size_t vl;
    vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
 
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
 
        vfloat32m1_t vx = __riscv_vle32_v_f32m1(x + i, vl);
        vfloat32m1_t vy = __riscv_vle32_v_f32m1(y + i, vl);
 
        vacc = __riscv_vfmacc_vf_f32m1(vacc, 1.0f,
               __riscv_vfmul_vv_f32m1(vx, vy, vl), vl);
    }
 
    return __riscv_vfmv_f_s_f32m1_f32(vacc);
}

float dot_rvv(const float *x, const float *y, size_t n) {
    size_t vl;
    /* Accumulator register: m1 group, zero-initialized */
    vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
 
    /* Use m8 grouping to maximise VLEN utilisation per strip */
    vfloat32m8_t vsum = __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());
 
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m8(n - i);
 
        vfloat32m8_t vx = __riscv_vle32_v_f32m8(x + i, vl);
        vfloat32m8_t vy = __riscv_vle32_v_f32m8(y + i, vl);
 
        /* Accumulate partial products into vsum */
        vsum = __riscv_vfmacc_vf_f32m8(vsum, 1.0f,
               __riscv_vfmul_vv_f32m8(vx, vy, vl), vl);
    }
 
    /* Reduce the m8 vector register group down to a scalar */
    vl = __riscv_vsetvlmax_e32m8();
    vacc = __riscv_vfredusum_vs_f32m8_f32m1(vsum, vacc, vl);
    return __riscv_vfmv_f_s_f32m1_f32(vacc);
}

int main() {
    const size_t n = 1 << 12; // Example vector length
    float x[n], y[n];
 
    // Initialize x and y with example values
    for (size_t i = 0; i < n; i++) {
        x[i] = (float)i;
        y[i] = (float)(n - i);
    }
    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    float result_scalar = dot_scalar(x, y, n);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif
    printf("Dot product (scalar): %f\n", result_scalar);

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    float result_rvv_naive = dot_rvv_naive(x, y, n);
    printf("Dot product (RVV naive): %f\n", result_rvv_naive);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif

    #ifdef GEM5
        m5_reset_stats(0, 0);
    #endif
    float result_rvv = dot_rvv(x, y, n);
    printf("Dot product (RVV): %f\n", result_rvv);
    #ifdef GEM5
        m5_dump_stats(0, 0);
    #endif


 
    return 0;
}