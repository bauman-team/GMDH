cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(GMDH VERSION 0.1)

add_executable(GMDH src/main.cpp
                    src/gmdh.h
                    src/gmdh.cpp)


add_subdirectory(external/armadillo-11.2.3)
target_link_libraries(${PROJECT_NAME} PUBLIC armadillo) 
target_include_directories(${PROJECT_NAME} PUBLIC external/armadillo-11.2.3/include)

set(BOOST_ROOT "/home/mikhail-xnor/Downloads/boost_1_79_0")
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.79.0 COMPONENTS system thread) 

if(Boost_FOUND)
    target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES})
endif()
