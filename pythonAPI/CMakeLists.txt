cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES ../src/common.h
    ../src/gmdh_objects.h
    ../src/gmdh.h
    ../src/gmdh.cpp
    ../src/criterion.h 
    ../src/criterion.cpp 
    ../src/combi.h 
    ../src/combi.cpp 
    ../src/multi.h 
    ../src/multi.cpp
    ../src/gmdh_lib.h
    ../src/mia.h 
    ../src/mia.cpp
	../src/ria.h
	../src/ria.cpp
    gmdh.cpp) 

pybind11_add_module(${MODULE_NAME}  ${SOURCES})
target_include_directories(${MODULE_NAME} PRIVATE ../external/pybind11-2.10.0/include)
target_include_directories(${MODULE_NAME} PRIVATE ../external/eigen-3.4)
target_include_directories(${MODULE_NAME} PRIVATE ../external/indicators-master/include)
target_compile_definitions(${MODULE_NAME} PRIVATE "GMDH_MODULE")


if(Boost_FOUND)
    target_include_directories(${MODULE_NAME} PRIVATE ${Boost_INCLUDE_DIR})
    target_link_libraries(${MODULE_NAME} PRIVATE ${Boost_LIBRARIES})
endif()