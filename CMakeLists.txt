cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(GMDH VERSION 0.1)

add_executable(GMDH src/main.cpp)

add_subdirectory(external/armadillo-11.2.3)
target_link_libraries(${PROJECT_NAME} PUBLIC armadillo) 
target_include_directories(${PROJECT_NAME} PUBLIC external/armadillo-11.2.3/include)