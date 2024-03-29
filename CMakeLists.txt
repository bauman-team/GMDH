cmake_minimum_required(VERSION 3.16)

project(GMDH)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE) # basic compilation on Release
  set(CMAKE_BUILD_TYPE Release)
endif()

if (CMAKE_HOST_WIN32)
    set(WINDOWS 1)
elseif(CMAKE_HOST_APPLE)
    set(MACOS 1)
elseif(CMAKE_HOST_UNIX)
    set(LINUX 1)
endif()

if (NOT WINDOWS) 
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-libgcc")
endif()

if (LINUX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
elseif (MACOS) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

find_package(Boost 1.79.0 QUIET COMPONENTS system thread filesystem json chrono)
if(NOT Boost_FOUND)
    set(Boost_USE_STATIC_LIBS OFF) 
    find_package(Boost 1.79.0 REQUIRED COMPONENTS system thread filesystem json chrono)
    message( WARNING "The library and module will get the required boost dependency! "
        "Not found static boost libs, and used shared libs!" )
endif()



set(LIBRARY_NAME gmdh)
set(MODULE_NAME _gmdh_core)
set(PROJECT_EXAMPLES gmdh_example) # only for examples

add_subdirectory(src/)
add_subdirectory(external/pybind11-2.10.0)
add_subdirectory(pythonAPI/)
add_subdirectory(examples/)
if(BUILD_TESTS MATCHES yes)
    if (WINDOWS)
	    add_subdirectory(external/gtest)
    endif()
    set(TESTS_NAME gmdh_tests)
    add_subdirectory(tests/)
endif()

set_target_properties(${LIBRARY_NAME} ${MODULE_NAME} ${TESTS_NAME} ${PROJECT_EXAMPLES}
    PROPERTIES
    #ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
)
