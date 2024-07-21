//
// Created by ybarak on 27/06/2024.
//
#include "MapReduceFramework.h"
#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <list>
#include <algorithm>
#include <pthread.h>
#include <unistd.h>

struct JobContext;
struct ThreadContext;

// helper functions
void* threadRun(void* _arg);
unsigned long getInputPairIndex(ThreadContext *tc);
unsigned long getInputPairIndex_after_reduce(ThreadContext *tc);
void processInputPair(ThreadContext *tc, unsigned long index);
void reducePair(ThreadContext *tc, IntermediateVec cur_pairs);
void executeShuffleOperation(JobContext* jobContext);
void sortIntermediatePairsByKeys(ThreadContext *threadCtx);
void check_destroy(int check_input);
void InitReduceStage(JobContext *jobContext);

/**
 * class Barrier - make a barrier before the shuffle and release the thread after finish the shuffle
 */
class Barrier {
public:
	Barrier(int numThreads);
    void barrier(JobContext *jobContext);
    ~Barrier();


private:
    pthread_mutex_t mutex;
    pthread_cond_t cv;
    int count;
    int numThreads;
};

/**
 *  struct that hold all the information in this job
 */
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

/**
 * struct for each thread that has his intermediateVec and  pointer to the job
 */
struct ThreadContext {
    int threadID;
    JobContext* jobContext;
    IntermediateVec intermediateVec;
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



void Barrier::barrier(JobContext* jobContext)
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
        executeShuffleOperation(jobContext);
        InitReduceStage(jobContext);

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
    IntermediatePair intermediatePair = std::make_pair(key, value);
    if (pthread_mutex_lock(&threadContext->jobContext->vectorMutex) != 0) {
        fprintf(stderr, "[emit2: Failed to lock vectorMutex before adding pair]");
        exit(EXIT_FAILURE);
    }
    threadContext->intermediateVec.push_back(intermediatePair);
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
    OutputPair newOutputPair = std::make_pair(key, value);;
    if (pthread_mutex_lock(&threadContext->jobContext->vectorMutex) != 0) {
        fprintf(stderr, "[emit3: Failed to lock vectorMutex before adding output pair]");
        exit(EXIT_FAILURE);
    }
    threadContext->jobContext->outputVec->push_back(newOutputPair);
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



void executeMapping(ThreadContext *threadContext) {
    threadContext->jobContext->jobState.stage = MAP_STAGE;
    unsigned long inputIndex = 0;
    unsigned long inputSize = threadContext->jobContext->inputVec->size();
    while (true) {
        inputIndex = getInputPairIndex(threadContext);
        if (inputIndex >= inputSize) {
            break;
        }
        processInputPair(threadContext, inputIndex);
    }
}

void executeReduce(ThreadContext *threadContext) {
    unsigned long OutputPairIndex = 0;
    while(OutputPairIndex < threadContext->jobContext->shuffleArray.size()){
        OutputPairIndex = getInputPairIndex(threadContext);
        if(OutputPairIndex >= threadContext->jobContext->shuffleArray.size()){
            break;
        }
        std::vector<std::vector<IntermediatePair>> cur_vec = threadContext->jobContext->shuffleArray;
        const IntermediateVec cur_pairs_vec = threadContext->jobContext->shuffleArray.at(OutputPairIndex);
        reducePair(threadContext, cur_pairs_vec);
    }
}

/**
 * the function of the theard that run the map and reduce
 * @param _arg - ThreadContext struct that belong to the thread
 */
void* threadRun(void* _arg){
    auto* threadContext = (ThreadContext*) _arg;

    executeMapping(threadContext);

    sortIntermediatePairsByKeys(threadContext);

    threadContext->jobContext->barrier->barrier(threadContext->jobContext);

    executeReduce(threadContext);
}

/**
 * Retrieves the next available index and increments the atomic counter.
 * @param tc - Pointer to the ThreadContext structure associated with the thread.
 * @return The next index in the array that the thread should process.
 */
unsigned long getInputPairIndex(ThreadContext *tc) {
    unsigned long mask = (1 << 31) - 1;
    unsigned long curValue = (tc->jobContext->counterAtomic->fetch_add(1, std::memory_order_relaxed));
    unsigned long nextValueIndex = curValue & mask; // first 31 bit - the num of unmapped values
    return nextValueIndex;
}


void atomicCounterHandler(ThreadContext *tc) {
    if (pthread_mutex_lock(&tc->jobContext->counterMutex) != 0) {
        fprintf(stderr, "[updateAtomicCounter] Error: Unable to lock counterMutex (code %d).\n");
        exit(EXIT_FAILURE);
    }

    unsigned long incrementValue = 0x80000000;
    tc->jobContext->counterAtomic->fetch_add(incrementValue, std::memory_order_relaxed);

    unsigned long counterValue = tc->jobContext->counterAtomic->load(std::memory_order_relaxed);
    unsigned long processedPairs = (counterValue >> 31) & 0x7fffffff;

    float completionRate = (static_cast<float>(processedPairs) / static_cast<float>(tc->jobContext->maxSize)) * 100.0f;
    tc->jobContext->jobState.percentage = completionRate;

    if ( pthread_mutex_unlock(&tc->jobContext->counterMutex) != 0) {
        fprintf(stderr, "[updateAtomicCounter] Error: Unable to unlock counterMutex (code %d).\n");
        exit(EXIT_FAILURE);
    }

}

/**
 * send the pair to the map func and add that one finish to the atomic counter
 * @param tc - ThreadContext struct that belong to the thread
 * @param cur_pair pair of key value that map function get
 */
void processInputPair(ThreadContext *tc, unsigned long i) {
    (*(tc->jobContext->mapReduceClient)).map((tc->jobContext->inputVec)->at(i).first, (tc->jobContext->inputVec)->at(i).second, tc);

    atomicCounterHandler(tc);
}

/**
 * send the pair to the reduce func and add that one finish to the atomic counter
 * @param tc - ThreadContext struct that belong to the thread
 * @param cur_pair pair of key value that reduce function get
 */
void reducePair(ThreadContext *tc, const IntermediateVec curPair) {
    (*(tc->jobContext->mapReduceClient)).reduce(&curPair, tc);
    atomicCounterHandler(tc);
}

void InitReduceStage(JobContext *jobContext) {
    jobContext->maxSize = jobContext->shuffleArray.size();
    jobContext->counterAtomic->store(0);
    jobContext->counterAtomic->fetch_add(0x8000000000000000);
    jobContext->counterAtomic->fetch_add(0x4000000000000000);
    jobContext->jobState = {REDUCE_STAGE, 0.0f};
}


/**
 * Configures the environment for the shuffle operation and resets progress counters.
 * @param jobDetails - Struct containing the state and data of the current job
 */
void configureShuffleEnvironment(JobContext *jobDetails) {
    if (pthread_mutex_lock(&jobDetails->stageMutex) != 0){
        perror("Failed to lock stage mutex at shuffle initialization");
        exit(EXIT_FAILURE);
    }

    // Initialize the counter and set the upper bit to mark start of shuffle
    jobDetails->counterAtomic->store(0x8000000000000000);

    // Update the job state to indicate the shuffle stage has begun
    jobDetails->jobState.stage = SHUFFLE_STAGE;
    jobDetails->jobState.percentage = 0;

    // Compute the total number of key-value pairs across all threads
    unsigned long pairCount = 0;
    for (int idx = 0; idx < jobDetails->multiThreadLevel; idx++) {
        pairCount += (jobDetails->threadContexts + idx)->intermediateVec.size();
    }
    jobDetails->maxSize = pairCount;

    // Adjust the counter to include the pair count at a specific bit position
    jobDetails->counterAtomic->fetch_add(pairCount << 31);

    if (pthread_mutex_unlock(&jobDetails->stageMutex) != 0){
        perror("Failed to unlock stage mutex after shuffle setup");
        exit(EXIT_FAILURE);
    }
}


/**
 * Helper function to find the largest key in the current intermediate vectors.
 * This version uses a direct reference to keep track of the largest pair, simplifying the logic.
 * @param jobContext - Contains all thread contexts and their vectors.
 * @return The largest intermediate pair found across all threads.
 */
IntermediatePair findLargestKey(JobContext* jobContext) {
    IntermediatePair* largestPair = nullptr;

    for (int i = 0; i < jobContext->multiThreadLevel; ++i) {
        IntermediateVec& vec = (jobContext->threadContexts + i)->intermediateVec;
        if (!vec.empty() && (!largestPair || *largestPair->first < *vec.back().first )) {
            largestPair = &vec.back();
        }
    }

    return largestPair ? *largestPair : IntermediatePair(nullptr, nullptr); // Return an empty pair if none found
}

/**
 * Collects all pairs with the same key as the provided largest pair.
 * This version uses iterators for clarity and better control over vector manipulation.
 * @param jobContext - Contains all thread contexts and their vectors.
 * @param largestPair - The key to match against.
 * @return A vector of all pairs matching the largest key.
 */
std::vector<IntermediatePair> collectPairsWithKey(JobContext* jobContext, const IntermediatePair& largestPair) {
    std::vector<IntermediatePair> collectedPairs;

    for (int i = 0; i < jobContext->multiThreadLevel; ++i) {
        IntermediateVec& vec = jobContext->threadContexts[i].intermediateVec;

        // Use a reverse iterator to efficiently pop from the back of the vector
        for (auto rit = vec.rbegin(); rit != vec.rend(); ) {
            if (!(*rit->first < *largestPair.first) && !(*largestPair.first < *rit->first)) {
                collectedPairs.push_back(*rit);
                rit = std::vector<IntermediatePair>::reverse_iterator(vec.erase((++rit).base())); // Erase and move iterator back to the next element
            } else {
                break; // Since we're assuming vec is sorted or we are looking for the largest key until mismatch
            }
        }
    }

    return collectedPairs;
}


/**
 * Updates the shuffle progress in the job context based on current state.
 * @param jobContext - Context containing job details and current progress.
 */
void updateShuffleProgress(JobContext* jobContext) {
    jobContext->jobState.percentage = static_cast<float>(jobContext->shuffleArray.size()) / jobContext->maxSize * 100.0f;
}

/**
 * Executes the shuffle operation, organizing data into shuffled arrays by key.
 * @param jobContext - struct that holds all the information in this job.
 */
void executeShuffleOperation(JobContext* jobContext) {
    configureShuffleEnvironment(jobContext);

    while (jobContext->jobState.percentage < 100) {
        IntermediatePair largestPair = findLargestKey(jobContext);
        if (!largestPair.first) { // No more pairs available, stop the shuffle
            break;
        }

        std::vector<IntermediatePair> matchedPairs = collectPairsWithKey(jobContext, largestPair);
        if (!matchedPairs.empty()) {
            jobContext->shuffleArray.push_back(matchedPairs);
            updateShuffleProgress(jobContext);
        }
    }
}



/**
 * Sorts the intermediate vector for the current thread based on keys
 * @param threadCtx - ThreadContext structure associated with the thread
 */
void sortIntermediatePairsByKeys(ThreadContext *threadCtx) {
    std::sort(threadCtx->intermediateVec.begin(),
              threadCtx->intermediateVec.end(),
              [](const IntermediatePair &a, const IntermediatePair &b) {
                  return *(a.first) < *(b.first);
              });
}


/**
 * hold the current running until the job will be finished.
 * we will split the behaviour into 3 options by waitFlag. (0 = join(wait) for all the threadHandles,
 *                                                           1 = wait for already waiting thread
 *                                                           2 = the job already finished - dont wait)
 * @param job struct that hold all the information in this job
 */
void waitForJob(JobHandle job){
    JobContext* cur_job = ((JobContext*) job);
    // job already finished
    if(cur_job->waitFlag == 2){
        return;
    }
    // first time we enter this function - wait for all the threadHandles to terminate
    if (cur_job->waitFlag == 0){
        cur_job->waitFlag = 1;
        for (int i = 0; i < cur_job->multiThreadLevel; i++) {
            pthread_join(cur_job->threadHandles[i], NULL);
        }
        if (pthread_cond_broadcast(&cur_job->conditionVar) != 0) {
            fprintf(stderr, "[[Barrier]] error on pthread_cond_broadcast");
            exit(1);
        }
        cur_job->waitFlag = 2;
    }
    // more than one time we enter this function and the job still processing.
    if (cur_job->waitFlag == 1){
        if (pthread_cond_wait(&cur_job->conditionVar, &cur_job->barrierMutex) != 0){
            fprintf(stderr, "[[Barrier]] error on pthread_cond_wait");
            exit(1);
        }
    }
}

/**
 * take the JobState from the struct JobContext all the function is in mutex
 * @param job - struct that hold all the information in this job
 * @param state - JobState to insert the job state
 */
void getJobState(JobHandle job, JobState* state){
    JobContext* cur_job = (JobContext*)job;
    if (pthread_mutex_lock(&cur_job->stageMutex) != 0){
        fprintf(stderr, "[[In getJobState]] error on pthread_mutex_lock");
        exit(1);
    }
    *state = ((JobContext*) job)->jobState;
    if (pthread_mutex_unlock(&cur_job->stageMutex) != 0){
        fprintf(stderr, "[[In getJobState]] error on pthread_mutex_unlock");
        exit(1);
    }

}

/**
 * delete all the memory and destroy all the mutex/cond
 * @param job  - struct that hold all the information in this job
 */
void closeJobHandle(JobHandle job){
    waitForJob(job);
    JobContext* cur_job = ((JobContext*) job);
    check_destroy(pthread_mutex_destroy(&cur_job->barrierMutex));
    check_destroy(pthread_mutex_destroy(&cur_job->stageMutex));
    check_destroy(pthread_mutex_destroy(&cur_job->counterMutex));
    check_destroy(pthread_mutex_destroy(&cur_job->vectorMutex));
    check_destroy(pthread_cond_destroy(&cur_job->conditionVar));
    delete[] cur_job->threadHandles;
    delete[] cur_job->threadContexts;
    delete cur_job->counterAtomic;
    delete cur_job->barrier;
    delete cur_job;
}

/**
 * check if the destroy of the mutex or cond work , if not print and exit
 * @param check_input - function to check
 */
void check_destroy(int check_input) {
    if (check_input != 0) {
        fprintf(stderr, "[[closeJobHandle]] error on pthread_mutex/cond_destroy");
        exit(1);
    }
}