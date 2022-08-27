cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_EXAMPLES} main.cpp)
target_include_directories(${PROJECT_EXAMPLES} PRIVATE ../src/ ../external/eigen-3.4 ../external/indicators-master/include)
target_link_libraries(${PROJECT_EXAMPLES} PRIVATE ${LIBRARY_NAME}) 

if(WIN32)
    target_compile_definitions(${PROJECT_EXAMPLES} PRIVATE "GMDH_LIB")
endif()

if(Boost_FOUND)
    target_include_directories(${PROJECT_EXAMPLES} PRIVATE ${Boost_INCLUDE_DIR})
    target_link_libraries(${PROJECT_EXAMPLES} PRIVATE ${Boost_LIBRARIES})
endif()

