cmake_minimum_required(VERSION 3.14)
project(mobileclip_classifier CXX)
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT (system installation)
include_directories(/usr/include/x86_64-linux-gnu)
link_directories(/usr/lib/x86_64-linux-gnu)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Common source files (exclude main.cpp and batch_main.cpp)
set(COMMON_SOURCES
    src/classifier.cpp
    src/logger.cpp
    src/image_preprocess.cpp
    src/image_encoder_trt.cpp
    src/text_preprocess.cpp
    src/text_encoder_trt.cpp
    src/tokenizer.cpp
)

# Single image classifier
add_executable(classifier 
    src/main.cpp
    ${COMMON_SOURCES}
)

# Batch traffic light classifier
add_executable(batch_traffic_light_classifier 
    src/batch_main.cpp
    ${COMMON_SOURCES}
)

# Link libraries for single classifier
target_link_libraries(classifier PRIVATE 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
)

# Link libraries for batch classifier
target_link_libraries(batch_traffic_light_classifier PRIVATE 
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    cudart
)
