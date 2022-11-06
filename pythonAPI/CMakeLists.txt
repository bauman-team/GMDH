cmake_minimum_required(VERSION 3.19)

if (LINUX)
    if (CMAKE_BUILD_TYPE MATCHES Coverage) # testing code coverage 
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    endif()
endif()

set(SOURCES ../src/common.h
    ../src/gmdh_objects.h
    ../src/gmdh.h
    ../src/gmdh.cpp
    ../src/criterion.h 
    ../src/criterion.cpp 
	../src/linear_model.h
	../src/linear_model.cpp
    ../src/combi.h 
    ../src/combi.cpp 
    ../src/gmdh_lib.h
    ../src/mia.h 
    ../src/mia.cpp
	../src/multi.h
	../src/multi.cpp
	../src/ria.h
	../src/ria.cpp
    gmdh.cpp) 

pybind11_add_module(${MODULE_NAME}  ${SOURCES})

target_compile_definitions(${MODULE_NAME} PRIVATE "GMDH_MODULE")

target_include_directories(${MODULE_NAME} PRIVATE ../external/eigen-3.4
    ../external/indicators-master/include
    ../external/pybind11-2.10.0/include
    ${Boost_INCLUDE_DIR})
    
target_link_libraries(${MODULE_NAME} PRIVATE ${Boost_LIBRARIES})