cmake_minimum_required(VERSION 3.19)

project(GMDH VERSION 0.2)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_HOST_WIN32)
    set(WINDOWS 1)
elseif(CMAKE_HOST_APPLE)
    set(MACOS 1)
elseif(CMAKE_HOST_UNIX)
    set(LINUX 1)
endif()

if(WINDOWS)
    set(Boost_USE_STATIC_LIBS ON) 
else()
    set(Boost_USE_STATIC_LIBS OFF) 
endif()

find_package(Boost 1.79.0 COMPONENTS system thread filesystem json chrono) 
if(Boost_FOUND)
else()
    message( FATAL_ERROR "Need install boost dependencies!" )
endif()

set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

set(LIBRARY_NAME gmdh)
set(MODULE_NAME _gmdh_core)
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