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
void reduce_next_pair(ThreadContext *tc, IntermediateVec cur_pairs);
void init_shuffle_data(JobContext *jobContext);
void shuffle_state(JobContext* jobContext);
void start_reduce_stage(JobContext *jobContext);
void sortIntermediatePairsByKeys(ThreadContext *threadCtx);
void check_destroy(int check_input);

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
        shuffle_state(jobContext);
        start_reduce_stage(jobContext);
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

/**
 * the function of the theard that run the map and reduce
 * @param _arg - ThreadContext struct that belong to the thread
 */
void* threadRun(void* _arg){
    auto* threadContext = (ThreadContext*) _arg;

    executeMapping(threadContext);

    sortIntermediatePairsByKeys(threadContext);

    threadContext->jobContext->barrier->barrier(threadContext->jobContext);

    //finish the shuffle state and start reduce state
    unsigned long OutputPairIndex = 0;
    unsigned long shuffle_len = (threadContext->jobContext->shuffleArray.size());
    while(OutputPairIndex < shuffle_len){
        //take the next index to take from vector
        OutputPairIndex = getInputPairIndex_after_reduce(threadContext);
        if(OutputPairIndex >= shuffle_len){
            break;
        }
        std::vector<std::vector<IntermediatePair>> cur_vec = threadContext->jobContext->shuffleArray;
        const IntermediateVec cur_pairs_vec = threadContext->jobContext->shuffleArray.at(OutputPairIndex);
        reduce_next_pair(threadContext, cur_pairs_vec);
    }
}

/**
 * Retrieves the next available index and increments the atomic counter.
 * @param tc - Pointer to the ThreadContext structure associated with the thread.
 * @return The next index in the array that the thread should process.
 */
unsigned long getInputPairIndex(ThreadContext *tc) {
    unsigned long mask = (1 << 31) - 1;
    unsigned long cur_value = (tc->jobContext->counterAtomic->fetch_add(1, std::memory_order_relaxed));
    unsigned long next_value_index = cur_value & mask; // first 31 bit - the num of unmapped values
    return next_value_index;
}


/**
 * take the next index and add 1 to the atomic counter
 * @param tc - ThreadContext struct that belong to the thread
 * @return the next index in the array that the thread need to work on
 */
unsigned long getInputPairIndex_after_reduce(ThreadContext *tc) {
    // get next valid input
    if (pthread_mutex_lock(&tc->jobContext->counterMutex) != 0){
        fprintf(stderr, "[[Before Reduce, Before Get Next Input]] error on pthread_mutex_lock");
        exit(1);
    }
    unsigned long mask;
    mask = (1 << 31) - 1;
    unsigned long cur_value = (tc->jobContext->counterAtomic->load());
    unsigned long next_value_index = cur_value & mask; // first 31 bit - the num of unmapped values
    (*(tc->jobContext->counterAtomic))++;
    if (pthread_mutex_unlock(&tc->jobContext->counterMutex) != 0) {
        fprintf(stderr, "[[Before Reduce, After Get Next Input]] error on pthread_mutex_unlock");
        exit(1);
    }
    return next_value_index;
}

/**
 * send the pair to the map func and add that one finish to the atomic counter
 * @param tc - ThreadContext struct that belong to the thread
 * @param cur_pair pair of key value that map function get
 */
void processInputPair(ThreadContext *tc, unsigned long i) {
    // use map on cur_pair
(*(tc->jobContext->mapReduceClient)).map((tc->jobContext->inputVec)->at(i).first, (tc->jobContext->inputVec)->at(i).second, tc);

    if (pthread_mutex_lock(&tc->jobContext->counterMutex) != 0){
        fprintf(stderr, "[[After Map, Before +1 processed pairs]] error on pthread_mutex_lock");
        exit(1);
    }
    //add one to the finish map pairs
    (*(tc->jobContext->counterAtomic)) += 0x80000000;  // + 8 equal 1000, 0 equal 0000, we have 3+4*7=31 zero
    unsigned long done_precessed = (tc->jobContext->counterAtomic)->load() >> 31 & (0x7fffffff);
    tc->jobContext->jobState.percentage = ((float)done_precessed) / (float)tc->jobContext->maxSize *100;
    if (pthread_mutex_unlock(&tc->jobContext->counterMutex) != 0){
        fprintf(stderr, "[[After Map, After +1 processed pairs]] error on pthread_mutex_lock");
        exit(1);
    }
}

/**
 * send the pair to the reduce func and add that one finish to the atomic counter
 * @param tc - ThreadContext struct that belong to the thread
 * @param cur_pair pair of key value that reduce function get
 */
void reduce_next_pair(ThreadContext *tc, const IntermediateVec cur_pairs) {
    // use map on cur_pair
    (*(tc->jobContext->mapReduceClient)).reduce(&cur_pairs, tc);
    if (pthread_mutex_lock(&tc->jobContext->counterMutex) != 0){
        fprintf(stderr, "[[After Reduce, Before +1 processed pairs]] error on pthread_mutex_lock");
        exit(1);
    }
    //add one to the finish reduce pairs
    (*(tc->jobContext->counterAtomic)) += 0x80000000;  // + 8 equal 1000, 0 equal 0000, we have 3+4*7=31 zero
    unsigned long done_precessed = (tc->jobContext->counterAtomic)->load() >> 31 & (0x7fffffff);
    tc->jobContext->jobState.percentage = ((float)done_precessed) / (float)tc->jobContext->maxSize * 100;
    if (pthread_mutex_unlock(&tc->jobContext->counterMutex) != 0){
        fprintf(stderr, "[[After Reduce, After +1 processed pairs]] error on pthread_mutex_lock");
        exit(1);
    }
}

/**
 * change the state to shuffle and init the atomic counter to 0 finish and
 * count all the pair and add in the 31 bit the count
 * @param jobContext - struct that hold all the information in this job
 */
void init_shuffle_data(JobContext *jobContext) {//change the state of job to shuffle
    if (pthread_mutex_lock(&jobContext->stageMutex) != 0){
        fprintf(stderr, "[[After Map, Before Shuffle]] error on pthread_mutex_lock");
        exit(1);
    }
    (jobContext->counterAtomic)->operator=(0);
    (*(jobContext->counterAtomic)) += 0x8000000000000000;  // + 8 equal 1000, 0 equal 0000, we have 3+4*15=63 zero
    jobContext->jobState = {SHUFFLE_STAGE, 0};
    unsigned long count = 0;
    for (int i = 0; i < jobContext->multiThreadLevel;i++){
        count += (jobContext->threadContexts + i)->intermediateVec.size();
    }
    jobContext->maxSize = count;
    (*(jobContext->counterAtomic)) += 0x80000000 * count;  // + 8 equal 1000, 0 equal 0000, we have 3+4*7=31 zero
    if (pthread_mutex_unlock(&jobContext->stageMutex) != 0){
        fprintf(stderr, "[[After Map, Before Shuffle]] error on pthread_mutex_unlock");
        exit(1);
    }
}

/**
 * in loop take the biggest key chack in ak the thread and add all the pair with this key
 * to vector and add the vector to the shuffleArray
 * @param jobContext - struct that hold all the information in this job
 */
void shuffle_state(JobContext* jobContext){
    std::vector<IntermediatePair> cur_array;
    float finish_count = 0;
    IntermediatePair biggest_pair;
    init_shuffle_data(jobContext);
    while (jobContext->jobState.percentage < 100) {
        //find same key to start
        for(int i = 0; i < jobContext->multiThreadLevel; i++){
            cur_array = jobContext->threadContexts[i].intermediateVec;
            if(cur_array.size() > 0){
                biggest_pair = cur_array.at(cur_array.size()-1);
                break;
            }
        }
        //take the biggest key
        for (int i = 0; i < jobContext->multiThreadLevel; i++) {
            cur_array = (jobContext->threadContexts + i)->intermediateVec;
            if (!cur_array.empty()){
                if (!(*cur_array.at(cur_array.size()-1).first < *biggest_pair.first)) {
                    biggest_pair = cur_array.at(cur_array.size() - 1);
                }
            }
        }
        //made vector for this key
        std::vector<IntermediatePair> key_biggest_vector;
        for (int i = 0; i < jobContext->multiThreadLevel; i++) {
            cur_array = (jobContext->threadContexts + i)->intermediateVec;
            //finish this array
            while (!cur_array.empty() && !(*cur_array.at(cur_array.size()-1).first < *biggest_pair.first) &&
            !(*biggest_pair.first < *cur_array.at(cur_array.size()-1).first)) {
                //check if the least one of all the thread vector is good for this vector
                bool check_eq = !(*cur_array.at(cur_array.size()-1).first < *biggest_pair.first) &&
                        !(*biggest_pair.first < *cur_array.at(cur_array.size()-1).first);
                if(check_eq){
                    //insert to the vector, delete, update the percentage
                    key_biggest_vector.push_back(cur_array.at(cur_array.size()-1));
                    cur_array.pop_back();
                    (jobContext->threadContexts + i)->intermediateVec.pop_back();
                    (*(jobContext->counterAtomic))++;
                    finish_count += 1;
                    jobContext->jobState.percentage = finish_count/(float) jobContext->maxSize * 100;
                }
            }
        }
        jobContext->shuffleArray.push_back(key_biggest_vector);
    }
}

/**
 *  change the state to reduce and init the atomic counter to 0
 * @param jobContext - struct that hold all the information in this job
 */
void start_reduce_stage(JobContext *jobContext){
    jobContext->maxSize = jobContext->shuffleArray.size();
    //change the state and counterAtomic to reduce
    if (pthread_mutex_lock(&jobContext->stageMutex) != 0){
        fprintf(stderr, "[[After Shuffle, Before Reduce]] error on pthread_mutex_lock");
        exit(1);
    }
    (jobContext->counterAtomic)->operator=(0);
    (*(jobContext->counterAtomic)) += 0x8000000000000000;  // + 8 equal 1000, 0 equal 0000, we have 3+4*15=63 zero
    (*(jobContext->counterAtomic)) += 0x4000000000000000;  // + 4 equal 0100, 0 equal 0000, we have 2+4*15=62 zero
    jobContext->jobState = {REDUCE_STAGE, 0.0f};
    if (pthread_mutex_unlock(&jobContext->stageMutex) != 0){
        fprintf(stderr, "[[After Shuffle, Before Reduce]] error on pthread_mutex_lock");
        exit(1);
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