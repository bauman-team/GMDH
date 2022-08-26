cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES gmdh.h
    gmdh.cpp
    criterion.h 
    criterion.cpp 
    combi.h 
    combi.cpp 
    multi.h 
    multi.cpp) 

add_library(${LIBRARY_NAME} SHARED ${SOURCES})
target_include_directories(${LIBRARY_NAME} PRIVATE ../external/eigen-3.4)
target_include_directories(${LIBRARY_NAME} PRIVATE ../external/indicators-master/include)

if(WIN32)
    target_sources(${LIBRARY_NAME} PRIVATE gmdh_lib.h dllmain.cpp)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE "GMDH_EXPORTS" "GMDH_LIB")
endif()


if(Boost_FOUND)
    target_include_directories(${LIBRARY_NAME} PRIVATE ${Boost_INCLUDE_DIR})
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${Boost_LIBRARIES})
endif()