/*
 Fo this exercise, we will test a small program that do some calculations, both using standard code,
 and to test to do the same using vectorized code.

 The arm processor has a normal instruction set that uses SISD as per Flynn´s taxonomy
 (Single Instruction Single Data) run by its instruction execution pipeline. 
 It has 31 registers of 64-bits width, that can be used either at its full width or smaller sizes,
 depending on the calcualtions to be done.
 
 Beside these standard instructions, it also has vecotrized instruction set that can use 128-bite wide registers
 to store data in, and then perform computiations on this vector. 

 the vector could as an example hold 4 parallel 32-bit wide floating numbers that can be used multiplying another set
 of 4 parallel 32-bit floats, thus giving a 4-folded performance boost.
*/

#include <stdio.h>
#include<stdlib.h>
#include <arm_neon.h>
#include <time.h>

// standard version of multiplication
void mult_std(float* a, float* b, float* r, int num)
{
    for (int i = 0; i < num; i++)
    {
        r[i] = a[i] * b[i];
    }
}
// vectorized version of multiplication
void mult_vect(float* a, float* b, float* r, int num)
{
        float32x4_t va, vb, vr;

        for (int i = 0; i < num; i += 4)
        {
            // load 4 floats from each array into vector registers
            va = vld1q_f32(&a[i]);
            vb = vld1q_f32(&b[i]);
            // perform vectorized multiplication
            vr = vmulq_f32(va, vb);
            // store the result back to the result array
            vst1q_f32(&r[i], vr);
        }
}

int main(int argc, char *argv[])
{
    int num = 100000000;
    float *a = (float*)aligned_alloc(16, num * sizeof(float));
    float *b = (float*)aligned_alloc(16, num * sizeof(float));
    float *r = (float*)aligned_alloc(16, num * sizeof(float));

    for (int i = 0; i < num; i++)
    {
        a[i] = (float)(i + 1);
        b[i] = (float)(i + 1);
    }

    struct timespec ts_start;
    struct timespec ts_end_1;
    struct timespec ts_end_2;

    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    mult_std(a, b, r, num);
    clock_gettime(CLOCK_MONOTONIC, &ts_end_1);
    mult_vect(a, b, r, num);
    clock_gettime(CLOCK_MONOTONIC, &ts_end_2);

    double duration_std = (ts_end_1.tv_sec - ts_start.tv_sec) + (ts_end_1.tv_nsec - ts_start.tv_nsec) * 1e-9;
    double duration_vect = (ts_end_2.tv_sec - ts_end_1.tv_sec) + (ts_end_2.tv_nsec - ts_end_1.tv_nsec) * 1e-9;

    printf("Standard multiplication time: %f seconds\n", duration_std);
    printf("Vectorized multiplication time: %f seconds\n", duration_vect);

    free(a);
    free(b);
    free(r);

    return 0;
}

/*  SHORT-DESCRIPTION:
    The code generates two arrays a and b of floats with length 100,000,000 and a result array r of the same lenghth.
    the purpose of the program is just to calculate r[i] = a[i] * b[i] for all indices in the array.
    there are two functions, mult_std that uses standard C-code and mult_vect that uses an intrinsic funcation that uses SIMD
    instructions inside the NEON vectorized instruction set.

    The intrinsic functions are just wrappers around the corresponding assembly instructions, making it easier to use them in C/C++ code.

    Using the linux function clock_gettime found in time.h, we can time the two functions and see how they perform
    relative to each other. Note that clock_gettime fills in a struct containing two variables, tv_sec & tv_nsec, thus
    the number of seconds, as well as nanoseconds that has elapsed since the clock started. 

    to calculate the duration of our two function calls, we can subtract values from two readings and calculate a floating point
    value with the number of elapsed seconds.

    NOTE: When compiling the code, out of the box the compiler generates quite slow code. By adding a compiler
    directive -O0, -O1, -O2 or -O3 we can enable various levels of optimization. (The -O0 is defualt)
*/

/*

Analyzing Performance (By using Nvidia Nsight)
-O0 : Standard multiplication time: Approx : 0.60 s
    : Vectorized multiplication time: Approx : 0.50ms
-O1 : Standard multiplication time: Approx : 0.30 s
    : Vectorized multiplication time: Approx : 0.15s
-O2 : Standard multiplication time: Approx : 0.26s
    : Vectorized multiplication time: Approx : 0.15s
-O3 : Standard multiplication time: Approx : 0.25s
    : Vectorized multiplication time: Approx : 0.14 s

*/
