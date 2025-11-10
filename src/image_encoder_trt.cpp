// image_encoder_trt.cpp - Image encoder TensorRT implementation
#include "image_encoder_trt.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

ImageEncoderTRT::ImageEncoderTRT() : stream_(nullptr) {}

ImageEncoderTRT::~ImageEncoderTRT() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (stream_) cudaStreamDestroy(stream_);
}

bool ImageEncoderTRT::build_engine_from_onnx(const std::string& onnx_path, const std::string& engine_path) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        std::cerr << "Failed to create builder" << std::endl;
        return false;
    }
    
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network) {
        std::cerr << "Failed to create network" << std::endl;
        return false;
    }
    
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        std::cerr << "Failed to create parser" << std::endl;
        return false;
    }
    
    // Parse ONNX file
    std::cout << "Parsing ONNX file: " << onnx_path << std::endl;
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    
    // Configure builder
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
    
    // Enable FP16 if supported
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled" << std::endl;
    }
    
    // Create optimization profile for dynamic input
    auto profile = builder->createOptimizationProfile();
    const char* input_tensor_name = input_name_.c_str();
    profile->setDimensions(input_tensor_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input_tensor_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 3, 224, 224});
    profile->setDimensions(input_tensor_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 3, 224, 224});
    config->addOptimizationProfile(profile);
    
    // Build engine
    std::cout << "Building TensorRT engine... This may take a while." << std::endl;
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        std::cerr << "Failed to build engine" << std::endl;
        return false;
    }
    
    // Save engine
    std::ofstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
        std::cerr << "Failed to open engine file for writing" << std::endl;
        return false;
    }
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    engine_file.close();
    std::cout << "Engine saved to: " << engine_path << std::endl;
    
    return true;
}

bool ImageEncoderTRT::init(const std::string& onnx_path, const std::string& engine_path) {
    // Check if engine file exists
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.good()) {
        std::cout << "Engine file not found, building from ONNX..." << std::endl;
        if (!build_engine_from_onnx(onnx_path, engine_path)) {
            return false;
        }
    }
    engine_file.close();
    
    // Load engine
    std::cout << "Loading TensorRT engine: " << engine_path << std::endl;
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file" << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();
    
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    // Set input shape for dynamic shapes
    nvinfer1::Dims4 input_dims{1, 3, 224, 224};
    context_->setInputShape(input_name_.c_str(), input_dims);
    
    // Get output dimensions
    auto output_dims = context_->getTensorShape(output_name_.c_str());
    
    input_size_ = 1 * 3 * 224 * 224;
    output_size_ = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size_ *= output_dims.d[i];
    }
    
    std::cout << "Input size: " << input_size_ << std::endl;
    std::cout << "Output size: " << output_size_ << std::endl;
    
    // Allocate GPU memory
    cudaMalloc(&d_input_, input_size_ * sizeof(float));
    cudaMalloc(&d_output_, output_size_ * sizeof(float));
    
    // Set tensor addresses
    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);
    
    // Create CUDA stream
    cudaStreamCreate(&stream_);
    
    // Allocate host output buffer
    host_output_.resize(output_size_);
    
    std::cout << "ImageEncoderTRT initialized successfully" << std::endl;
    return true;
}

std::vector<float> ImageEncoderTRT::encode(const std::vector<float>& nchw_data, int batch, int channels, int height, int width) {
    // Copy input to GPU
    cudaMemcpyAsync(d_input_, nchw_data.data(), 
                   input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream_);
    
    // Execute inference
    context_->enqueueV3(stream_);
    
    // Copy output to host
    cudaMemcpyAsync(host_output_.data(), d_output_, 
                   output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    
    // Synchronize stream
    cudaStreamSynchronize(stream_);
    
    // Normalize
    std::vector<float> features(host_output_.begin(), host_output_.end());
    l2_normalize(features);
    
    return features;
}

void ImageEncoderTRT::l2_normalize(std::vector<float>& vec) {
    float norm = 0.0f;
    for (float v : vec) {
        norm += v * v;
    }
    norm = std::sqrt(norm);
    if (norm > 1e-12f) {
        for (float& v : vec) {
            v /= norm;
        }
    }
}
