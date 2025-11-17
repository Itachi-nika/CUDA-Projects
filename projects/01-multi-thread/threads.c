/* Below is a small program demonstrating a multi-threaded program.
    In main it launches 5 threads and then wats until all the threds have exited using pthread_join,
    before main itself returns. As you can see, one has to include pthread.h to access the standard POSIX
    functions to create threads. In other programming languages there are other methods to launch threads.
    for this instance we use C-code
*/




#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// Number of threads to create
#define NUM_THREADS 5

//Thread Function
void* printHello(void* threadid) {
    long tid = (long)threadid;
    printf("Hello from thread %ld\n", tid);
    pthread_exit(NULL);
    return NULL; //this line is typically not reached due to pthread_exit above
}

int main() 
{
    pthread_t threads[NUM_THREADS];
    int rc;
    for (long t = 0; t < NUM_THREADS; t++) {
        printf("Creating thread %ld\n", t);
        rc = pthread_create(&threads[t], NULL, printHello, (void*)t);
        if (rc) {
            printf("Error: Unable to create thread %ld, %d\n", t, rc);
            exit(-1);
        }
    }

    // Join the threads
    for (long t = 0; t < NUM_THREADS; t++) {
        pthread_join(threads[t], NULL);
    }

    printf("Main thread completing\n");

    return 0;
}

/*  pthread_create takes as first argument a pointer to a pthread_t object, second argument is thread attribute(unsued, thus NULL),
third argument is out thread function and fourth argument is a parameter sent to the thread. 
this parameter can for instance be used to give each thread a unique id. */

/* -------- QUESTIONS -----------*/

/* 
1. What is the primary purpose of pthread_create() ?
 Answer : To create a new thread.

2. How many threads, including the main thread, are running in this program?
 Answer : 6 threads (5 created threads + 1 main thread)

3. Which of the following, best describes the order of thread completion in this program?
    Answer : 
 
4. What is the role of pthread_join() in this program?
 Answer : To wait for a specific thread to complete before proceeding.
    it ensures that main thread waits for the worker threads to finish.

5. In the printHello function, what does pthread_exit(NULL) do?
 Answer : It terminates the current thread. 


*/