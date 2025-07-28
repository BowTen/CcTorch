# MNIST MLP example
cmake_minimum_required(VERSION 3.10)
project(MNISTExample)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the parent CcTorch library
set(CCTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${CCTORCH_ROOT}/include)

# Add the CcTorch library
add_library(cctorch STATIC
    ${CCTORCH_ROOT}/src/tensor.cc
    ${CCTORCH_ROOT}/src/layer.cc
    ${CCTORCH_ROOT}/src/mnist_loader.cc
)

# Create executable for MNIST MLP example
add_executable(mnist_mlp mlp_mnist.cc)

# Link library to executable
target_link_libraries(mnist_mlp cctorch)
