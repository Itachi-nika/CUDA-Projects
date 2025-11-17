/*  
    Our first steps into the world of parallel programming. It might not be very fance or exotic,
    but to be able to run, we first have to take some baby steps.
*/

#include <iostream>
#include <math.h>

__global__ void multKernel(int n, float* a, float* b, float* c)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] * b[i];
    }
}

int main()
{
    int N = 1<<20;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];

    // Allocate device memory
    cudaMalloc(&d_a, N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));

    // Initialize host Data
    for (int i = 0; i < N; i++)
    {
        h_a[i] = 2.0f;
        h_b[i] = 3.0f;
    }

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel
    multKernel<<<1, 1>>>(N, d_a, d_b, d_c);

    // Copy results from device to host
    cudaMemcpy(h_c, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Check results for errors (All values should be 6.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(h_c[i]-6.0f));
        std::cout << "Max error: " << maxError << std::endl;
    }

    //Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;

}

/*
    To compile this we use CUDA compiler nvcc instead of gcc
    (if you run it using $ ./ex1 not much happends, the program just multiply two vectors containing about 64 million values)
    We want to examine what happends a bit more using Nsys Tool: ($ nsys nvprof ./ex1)

    -- QUESTIONS --
    1. How much time is spent running the kernel, copying data from host to device and from device to host?
    Answer: Approx 10 second(s)
    Note: Not that impressive to spend so long time multiplying 64 million values!!
    Why is the performance so poor?
    Hint: Think about how many threads we are using to run the kernel.
    Answer: Because we are only using one thread to do all the work, we are not utilizing the parallel processing power of the GPU.
*/