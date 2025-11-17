## Introduction ##
In these assigments, we will focus on using the Jetson Orin Nano for high-performance computing, particulary using threads and some vectorized instructions to achieve high performance.

Pthreads provide a method to create multi-threaded applications, a key to harnessing the power of multi-core CPUs.

The vectorized instruction set (NEON) gives the possibility to run SIMD instructions and thus increase the computation
##                  ##

## What are the learning outcomes? ##

- Understand the basics of Jetson Orin Nano.

- Setup and run a simple Pthreads program on the Jetson Orin Nano.

- Understand the importance of multi-threaded programs in high-performance computing.

- See how vectorized code can speed up calculations by comparing scalar vs vectorized implementations.

-- EXTRA : inspect the generated assembly code and the the impact of the compiler optimization settings.

## SETUP ##
1. Hardware Requirements 
    - NVIDIA Jetson Orin Nano Super Developer Kit
    - MicroSD Card, 64 GB
    - trivial : Monitor, Keyboard, Mouse & Ethernet Cable(for internet connection)

2. Software Setup:
    - Jetson Nano Dev Kit SD Card Image for NVIDIA´s official site
    - Flashing the SD card using Etcher

3. Development Enviroment
    - Linux (ubento)
    - Nvidia Nsight
    - Vscode
    #Package List#
    - sudo apt update
    - sudo apt upgrade
    - sudo apt install gcc g++ make
##                                  ##