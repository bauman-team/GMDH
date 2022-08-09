cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(GMDH VERSION 0.1)

add_executable(GMDH src/main.cpp
                    src/gmdh.h
                    src/gmdh.cpp
                    src/criterion.h 
                    src/criterion.cpp 
                    src/combi.h 
                    src/combi.cpp)

set(LIBRARY_NAME gmdhpy)

add_subdirectory(external/pybind11-2.10.0)
pybind11_add_module(${LIBRARY_NAME} pythonAPI/gmdh.cpp
                                   src/gmdh.h
                                   src/gmdh.cpp
                                   src/criterion.h 
                                   src/criterion.cpp 
                                   src/combi.h 
                                   src/combi.cpp)
target_include_directories(${LIBRARY_NAME} PUBLIC external/pybind11-2.10.0/include)

target_include_directories(${PROJECT_NAME} PUBLIC external/eigen-3.4)
target_include_directories(${LIBRARY_NAME} PUBLIC external/eigen-3.4)

add_subdirectory(external/indicators-master)
target_include_directories(${PROJECT_NAME} PUBLIC external/indicators-master/include)
target_include_directories(${LIBRARY_NAME} PUBLIC external/indicators-master/include)


if(WIN32)
    set(BOOST_ROOT "C:/boost_1_79_0")
    set(Boost_USE_STATIC_LIBS ON) 
else()
    set(BOOST_ROOT "/home/mikhail-xnor/Downloads/boost_1_79_0") 
    set(Boost_USE_STATIC_LIBS OFF) 
endif()

set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.79.0 COMPONENTS system thread) 

if(Boost_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES})
    target_include_directories(${LIBRARY_NAME} PUBLIC ${Boost_INCLUDE_DIR})
    target_link_libraries(${LIBRARY_NAME} PUBLIC ${Boost_LIBRARIES})
endif()
