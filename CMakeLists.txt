cmake_minimum_required(VERSION 3.22)
project(EX3_OS)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)
include_directories(Barrier)

add_executable(EX3_OS
#        Atomic/atomicdemo.cpp
#        Atomic/atomicdemo_multivar.cpp
#        MapReduceFramework.cpp
#                Barrier/Barrier.cpp
#        Barrier/Barrier.h
#        Barrier/barrierdemo.cpp
#        MapReduceClient.h
#        MapReduceFramework.h
#        MapReduceFramework.h
        "Sample Client/sampleClient.cpp"
#        MapReduceFramework.cpp
        ddd.cpp
)
