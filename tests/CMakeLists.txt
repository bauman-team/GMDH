cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${TESTS_NAME} test_combi.cpp)
target_include_directories(${TESTS_NAME} PRIVATE ../src/ ../external/eigen-3.4 ../external/indicators-master/include)

if (WIN32)
	include(FetchContent)
	FetchContent_Declare(
	  googletest
	  URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip)
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)
endif()

target_link_libraries(${TESTS_NAME} PRIVATE ${LIBRARY_NAME})
target_link_libraries(${TESTS_NAME} PRIVATE gtest gtest_main)

if(Boost_FOUND)
    target_include_directories(${TESTS_NAME} PRIVATE ${Boost_INCLUDE_DIR}) # fix boost not find
    target_link_libraries(${TESTS_NAME} PRIVATE ${Boost_LIBRARIES})
endif()