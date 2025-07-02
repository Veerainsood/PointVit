cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(voxelizer LANGUAGES CXX CUDA)

# Ensure CUDA is found

find_package(CUDA REQUIRED)

# Find PyTorch

# Torch_DIR may need to be set to your PyTorch install path

find_package(Torch REQUIRED)

# Find pybind11

find_package(pybind11 REQUIRED)

# Set C++ standard

target_compile_features(\${PROJECT_NAME} PRIVATE cxx_std_14)

# Include directories

include_directories(
\${CMAKE_CURRENT_SOURCE_DIR}/include
\${torch_INCLUDE_DIRS}
)

# Source files

torch_module_sources(
SRC
src/voxelize.cpp
src/bindings.cpp
src/compute_keys.cu
src/fill_voxels.cu
)

# Define the Python extension module

torch_module(
NAME voxelizer
SOURCES \${SRC}
CUDA_NVCC_FLAGS "-O3"
)

# Link libraries

target_link_libraries(voxelizer PRIVATE
\${TORCH_LIBRARIES}
pybind11::module
)

# Set properties for CUDA

set_target_properties(voxelizer PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)

# Disable warnings for external libraries

target_compile_options(voxelizer PRIVATE
\$<\$\<COMPILE_LANGUAGE\:CUDA>:-Xcompiler -w>
)
