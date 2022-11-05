cmake_minimum_required(VERSION 3.19)

set(SOURCES main.cpp)

add_executable(${PROJECT_EXAMPLES} ${SOURCES})

if(WINDOWS)
    target_compile_definitions(${PROJECT_EXAMPLES} PRIVATE "GMDH_LIB")
endif()

target_include_directories(${PROJECT_EXAMPLES} PRIVATE ../src/ 
    ../external/eigen-3.4 
    ../external/indicators-master/include  
    ${Boost_INCLUDE_DIR})

target_link_libraries(${PROJECT_EXAMPLES} PRIVATE ${LIBRARY_NAME} 
    ${Boost_LIBRARIES}) 

