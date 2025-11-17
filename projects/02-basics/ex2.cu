/*  
    Lets see how we step by step can improve the performance, relative to ex1.cu
    As we said, we only used one of all the lovely cores to do the calculations. Lets change that!

    With a few changes to the kernel launch configuration and the kernel code we can utilize many threads in parallel.
    This will speed up the execution time significantly.

    Copying the code from ex1.cu and modifying it slightly, we can imporove the performance.

    Changes are commented in the code.

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
    multKernel<<<1, 256>>>(N, d_a, d_b, d_c);       // Changed to use 256 threads 

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