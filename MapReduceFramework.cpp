#include "MapReduceFramework.h"
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <list>
#include <algorithm>
#include <pthread.h>
#include <unistd.h>


/**
 * struct for each thread that has his intermediateVec and  pointer to the job
 */
struct ThreadContext {
    int threadID;
    JobContext* jobContext;
    std::vector<IntermediatePair> intermediateVec;  // READY order of threadHandles
};
struct JobContext{
    std::vector<std::vector<IntermediatePair>> shuffleArray;
    std::atomic<uint64_t>* counterAtomic;
    int multiThreadLevel;
    int waitFlag;
    unsigned long maxSize;
    const InputVec* inputVec;
    const MapReduceClient* mapReduceClient;
    Barrier* barrier;
    ThreadContext* threadContexts;
    JobState jobState;
    OutputVec* outputVec;
    pthread_t* threadHandles;
    pthread_mutex_t barrierMutex;
    pthread_mutex_t vectorMutex;
    pthread_mutex_t counterMutex;
    pthread_mutex_t stageMutex;
    pthread_cond_t conditionVar;
};

Barrier::Barrier(int numThreads)
        : mutex(PTHREAD_MUTEX_INITIALIZER)
        , cv(PTHREAD_COND_INITIALIZER)
        , count(0)
        , numThreads(numThreads)
{ }


Barrier::~Barrier()
{
    if (pthread_mutex_destroy(&mutex) != 0) {
        fprintf(stderr, "[[Barrier]] error on pthread_mutex_destroy");
        exit(1);
    }
    if (pthread_cond_destroy(&cv) != 0){
        fprintf(stderr, "[[Barrier]] error on pthread_cond_destroy");
        exit(1);
    }
}


void Barrier::barrier()
{
    if (pthread_mutex_lock(&mutex) != 0){
        fprintf(stderr, "[[Barrier]] error on pthread_mutex_lock");
        exit(1);
    }
    if (++count < numThreads) {
        if (pthread_cond_wait(&cv, &mutex) != 0){
            fprintf(stderr, "[[Barrier]] error on pthread_cond_wait");
            exit(1);
        }
    } else {
        count = 0;
        if (pthread_cond_broadcast(&cv) != 0) {
            fprintf(stderr, "[[Barrier]] error on pthread_cond_broadcast");
            exit(1);
        }
    }
    if (pthread_mutex_unlock(&mutex) != 0) {
        fprintf(stderr, "[[Barrier]] error on pthread_mutex_unlock");
        exit(1);
    }
}

/**
 * Inserts a key-value pair into the intermediate array of the calling thread.
 * The operation is protected by a mutex to ensure thread safety.
 * @param key - the key part of the intermediate pair.
 * @param value - the value part of the intermediate pair.
 * @param context - the context structure of the calling thread, which contains the intermediate array.
 */
void emit2(K2* key, V2* value, void* context) {
    auto* threadContext = static_cast<ThreadContext*>(context);

    // Create a new IntermediatePair
    IntermediatePair intermediatePair = std::make_pair(key, value);

    // Add the new key to the intermediateArray of the thread
    if (pthread_mutex_lock(&threadContext->jobContext->vectorMutex) != 0) {
        fprintf(stderr, "[emit2: Failed to lock vectorMutex before adding pair]");
        exit(EXIT_FAILURE);
    }

    threadContext->intermediateVec.push_back(intermediatePair); // Add new value to array of mapped values of this thread

    if (pthread_mutex_unlock(&threadContext->jobContext->vectorMutex) != 0) {
        fprintf(stderr, "[emit2: Failed to unlock vectorMutex after adding pair]");
        exit(EXIT_FAILURE);
    }
}


/**
 * Adds a key-value pair to the output array of the thread that calls this function.
 * The operation is performed within a mutex lock to ensure thread safety.
 * @param key - the key component of the output pair.
 * @param value - the value component of the output pair.
 * @param context - the context of the calling thread, which includes the output array.
 */
void emit3(K3* key, V3* value, void* context) {
    auto* threadContext = static_cast<ThreadContext*>(context);

    // Create a new OutputPair
    OutputPair newOutputPair = std::make_pair(key, value);;

    // Add the new pair to the output vector of the job context
    if (pthread_mutex_lock(&threadContext->jobContext->vectorMutex) != 0) {
        fprintf(stderr, "[emit3: Failed to lock vectorMutex before adding output pair]");
        exit(EXIT_FAILURE);
    }

    threadContext->jobContext->outputVec->push_back(newOutputPair); // Add new value to output vector

    if (pthread_mutex_unlock(&threadContext->jobContext->vectorMutex) != 0) {
        fprintf(stderr, "[emit3: Failed to unlock vectorMutex after adding output pair]");
        exit(EXIT_FAILURE);
    }
}

/**
 * Initializes the JobContext structure and creates all the threads.
 * @param client - the client that provides the input vector and the map and reduce functions.
 * @param inputVec - the input vector containing the data to be processed by the map-reduce algorithm.
 * @param outputVec - the vector that will hold the final values after the map and reduce stages.
 * @param multiThreadLevel - the number of threads to create.
 * @return - a struct that holds all the information for this job.
 */
JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel) {
    // Allocate resources for job context
    auto* barrier = new Barrier(multiThreadLevel);
    auto* threads = new pthread_t[multiThreadLevel];
    auto* atomicCounter = new std::atomic<uint64_t>(0);
    auto* threadContexts = new ThreadContext[multiThreadLevel];
    auto* jobContext = new JobContext;

    // Initialize thread contexts
    for (int i = 0; i < multiThreadLevel; ++i) {
        threadContexts[i] = {i, jobContext};
    }

    // Set job context fields
    jobContext->threadContexts = threadContexts;
    jobContext->multiThreadLevel = multiThreadLevel;
    jobContext->barrier = barrier;
    jobContext->counterAtomic = atomicCounter;
    jobContext->inputVec = &inputVec;
    jobContext->outputVec = &outputVec;
    jobContext->mapReduceClient = &client;
    jobContext->threadHandles = threads;
    jobContext->waitFlag = 0;
    jobContext->conditionVar = PTHREAD_COND_INITIALIZER;
    jobContext->barrierMutex = PTHREAD_MUTEX_INITIALIZER;
    jobContext->vectorMutex = PTHREAD_MUTEX_INITIALIZER;
    jobContext->stageMutex = PTHREAD_MUTEX_INITIALIZER;
    jobContext->counterMutex = PTHREAD_MUTEX_INITIALIZER;
    jobContext->maxSize = inputVec.size();
    jobContext->jobState = {UNDEFINED_STAGE, 0};

    // Create threads
    for (int i = 0; i < multiThreadLevel; ++i) {
        pthread_create(&threads[i], nullptr, threadRun, &threadContexts[i]);
    }

    return jobContext;
}