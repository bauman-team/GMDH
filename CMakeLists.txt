cmake_minimum_required(VERSION 3.19)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(GMDH VERSION 0.2)

if(WIN32)
    set(BOOST_ROOT "C:/boost_1_79_0")
    set(Boost_USE_STATIC_LIBS ON) 
else()
    set(BOOST_ROOT "/home/mikhail-xnor/Downloads/boost_1_79_0") 
    set(Boost_USE_STATIC_LIBS OFF) 
endif()

set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.79.0 COMPONENTS system thread filesystem) 

set(LIBRARY_NAME gmdh)
set(MODULE_NAME gmdhpy)
set(TESTS_NAME gmdh_tests)
set(PROJECT_EXAMPLES gmdh_example) # only for examples

add_subdirectory(src/)
add_subdirectory(external/pybind11-2.10.0)
add_subdirectory(pythonAPI/)
add_subdirectory(examples/)
add_subdirectory(tests/)

set_target_properties(${LIBRARY_NAME} ${MODULE_NAME} ${TESTS_NAME} ${PROJECT_EXAMPLES}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

enable_testing()
add_test(NAME ${TESTS_NAME}
	COMMAND ${TESTS_NAME})