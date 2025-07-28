cmake_minimum_required(VERSION 3.10)
project(CcTorch)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Add source files
set(SOURCES
    src/tensor.cc
    src/layer.cc
    src/mnist_loader.cc
)

# Create library
add_library(cctorch ${SOURCES})

# Install the library and headers for use by examples
install(TARGETS cctorch DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
