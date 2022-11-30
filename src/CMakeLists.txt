cmake_minimum_required(VERSION 3.16)

if (LINUX)
    if (CMAKE_BUILD_TYPE MATCHES Coverage) # testing code coverage 
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    endif()
endif()

set(SOURCES common.h
    gmdh_objects.h
    gmdh.h
    gmdh.cpp
    criterion.h 
    criterion.cpp 
	linear_model.h
	linear_model.cpp
    combi.h 
    combi.cpp 
	multi.h
	multi.cpp
	mia.h 
	mia.cpp
	ria.h
	ria.cpp) 

add_library(${LIBRARY_NAME} SHARED ${SOURCES})

if(WINDOWS)
    target_sources(${LIBRARY_NAME} PRIVATE gmdh_lib.h dllmain.cpp)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE "GMDH_EXPORTS") # for windows shaed lib __declspec(dllimport)
endif()

target_compile_definitions(${LIBRARY_NAME} PRIVATE "GMDH_LIB")

target_include_directories(${LIBRARY_NAME} PRIVATE ../external/eigen-3.4
    ../external/indicators-master/include
    ${Boost_INCLUDE_DIR})

target_link_libraries(${LIBRARY_NAME} PRIVATE ${Boost_LIBRARIES})
