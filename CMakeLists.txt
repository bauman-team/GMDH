cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(GMDH VERSION 0.2)

#file(GLOB LIBRARY_SOURCES src/*) # not good style

set(LIBRARY_SOURCES src/gmdh.h
    src/gmdh.cpp
    src/criterion.h 
    src/criterion.cpp 
    src/combi.h 
    src/combi.cpp 
    src/multi.h 
    src/multi.cpp
	src/mia.h 
	src/mia.cpp) 

set(LIBRARY_NAME gmdh)
add_library(${LIBRARY_NAME} SHARED ${LIBRARY_SOURCES})

set(MODULE_NAME gmdhpy)
add_subdirectory(external/pybind11-2.10.0)
pybind11_add_module(${MODULE_NAME}  ${LIBRARY_SOURCES} pythonAPI/gmdh.cpp)
target_include_directories(${MODULE_NAME} PRIVATE external/pybind11-2.10.0/include)

set(PROJECT_EXAMPLES gmdh_example) # only for examples
add_executable(${PROJECT_EXAMPLES} examples/main.cpp)
target_include_directories(${PROJECT_EXAMPLES} PRIVATE src/ external/eigen-3.4 external/indicators-master/include)
target_link_libraries(${PROJECT_EXAMPLES} PRIVATE ${LIBRARY_NAME}) 

if(WIN32)
    set(BOOST_ROOT "C:/boost_1_79_0")
    set(Boost_USE_STATIC_LIBS ON) 
    target_sources(${LIBRARY_NAME} PRIVATE src/gmdh_lib.h src/dllmain.cpp)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE "GMDH_EXPORTS" "GMDH_LIB")
    target_compile_definitions(${PROJECT_EXAMPLES} PRIVATE "GMDH_LIB")
else()
    set(BOOST_ROOT "/home/mikhail-xnor/Downloads/boost_1_79_0") 
    set(Boost_USE_STATIC_LIBS OFF) 
endif()

target_include_directories(${LIBRARY_NAME} PRIVATE external/eigen-3.4)
target_include_directories(${MODULE_NAME} PRIVATE external/eigen-3.4)

target_include_directories(${LIBRARY_NAME} PRIVATE external/indicators-master/include)
target_include_directories(${MODULE_NAME} PRIVATE external/indicators-master/include)

set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.79.0 COMPONENTS system thread) 

if(Boost_FOUND)
    target_include_directories(${LIBRARY_NAME} PRIVATE ${Boost_INCLUDE_DIR})
    target_link_libraries(${LIBRARY_NAME} PRIVATE ${Boost_LIBRARIES})
    target_include_directories(${MODULE_NAME} PRIVATE ${Boost_INCLUDE_DIR})
    target_link_libraries(${MODULE_NAME} PRIVATE ${Boost_LIBRARIES})
    target_include_directories(${PROJECT_EXAMPLES} PRIVATE ${Boost_INCLUDE_DIR})
    target_link_libraries(${PROJECT_EXAMPLES} PRIVATE ${Boost_LIBRARIES})
endif()


set(TESTS_NAME gmdh_tests)
add_executable(${TESTS_NAME}
	tests/test_combi.cpp
)
target_include_directories(${TESTS_NAME} PRIVATE src/ external/eigen-3.4 external/indicators-master/include)

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
target_include_directories(${TESTS_NAME} PRIVATE ${Boost_INCLUDE_DIR}) # fix boost not find
target_link_libraries(${TESTS_NAME} PRIVATE ${Boost_LIBRARIES})


enable_testing()
add_test(NAME ${TESTS_NAME}
	COMMAND ${TESTS_NAME})