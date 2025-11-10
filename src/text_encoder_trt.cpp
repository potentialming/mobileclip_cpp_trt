// text_encoder_trt.cpp - Text encoder TensorRT implementation
#include "text_encoder_trt.hpp"
#include <fstream>
#include <iostream>
#include <cmath>

TextEncoderTRT::TextEncoderTRT() : stream_(nullptr) {}

TextEncoderTRT::~TextEncoderTRT() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (stream_) cudaStreamDestroy(stream_);
}

bool TextEncoderTRT::build_engine_from_onnx(const std::string& onnx_path, const std::string& engine_path) {
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
    
    std::cout << "Parsing ONNX file: " << onnx_path << std::endl;
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled" << std::endl;
    }
    
    auto profile = builder->createOptimizationProfile();
    const char* input_tensor_name = input_name_.c_str();
    profile->setDimensions(input_tensor_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{1, max_seq_len_});
    profile->setDimensions(input_tensor_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{4, max_seq_len_});
    profile->setDimensions(input_tensor_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{16, max_seq_len_});
    config->addOptimizationProfile(profile);
    
    std::cout << "Building TensorRT engine... This may take a while." << std::endl;
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        std::cerr << "Failed to build engine" << std::endl;
        return false;
    }
    
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

bool TextEncoderTRT::init(const std::string& onnx_path, const std::string& engine_path) {
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.good()) {
        std::cout << "Engine file not found, building from ONNX..." << std::endl;
        if (!build_engine_from_onnx(onnx_path, engine_path)) {
            return false;
        }
    }
    engine_file.close();
    
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
    
    cudaStreamCreate(&stream_);
    
    std::cout << "TextEncoderTRT initialized successfully" << std::endl;
    return true;
}

std::vector<float> TextEncoderTRT::encode(const std::vector<int64_t>& input_ids) {
    // Encode batch of 1 and return the first (only) result
    auto batch_result = encode_batch(std::vector<std::vector<int64_t>>{input_ids});
    return batch_result[0];
}

std::vector<std::vector<float>> TextEncoderTRT::encode_batch(const std::vector<std::vector<int64_t>>& input_ids_batch) {
    int batch = input_ids_batch.size();
    
    nvinfer1::Dims2 input_dims{batch, max_seq_len_};
    context_->setInputShape(input_name_.c_str(), input_dims);
    
    auto output_dims = context_->getTensorShape(output_name_.c_str());
    
    size_t input_size = batch * max_seq_len_;
    size_t output_size = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size *= output_dims.d[i];
    }
    
    // Calculate embedding dimension per sample
    size_t emb_dim = output_size / batch;
    
    std::vector<int64_t> input_data(input_size, 0);
    for (int b = 0; b < batch; ++b) {
        const auto& ids = input_ids_batch[b];
        for (size_t i = 0; i < ids.size() && i < max_seq_len_; ++i) {
            input_data[b * max_seq_len_ + i] = ids[i];
        }
    }
    
    void* d_input_temp = nullptr;
    void* d_output_temp = nullptr;
    cudaMalloc(&d_input_temp, input_size * sizeof(int64_t));
    cudaMalloc(&d_output_temp, output_size * sizeof(float));
    
    cudaMemcpyAsync(d_input_temp, input_data.data(), 
                   input_size * sizeof(int64_t), cudaMemcpyHostToDevice, stream_);
    
    context_->setTensorAddress(input_name_.c_str(), d_input_temp);
    context_->setTensorAddress(output_name_.c_str(), d_output_temp);
    
    context_->enqueueV3(stream_);
    
    std::vector<float> host_output(output_size);
    cudaMemcpyAsync(host_output.data(), d_output_temp, 
                   output_size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    
    cudaStreamSynchronize(stream_);
    
    cudaFree(d_input_temp);
    cudaFree(d_output_temp);
    
    // Split and normalize each embedding separately
    std::vector<std::vector<float>> batch_embeddings;
    batch_embeddings.reserve(batch);
    
    for (int b = 0; b < batch; ++b) {
        std::vector<float> emb(host_output.begin() + b * emb_dim, 
                               host_output.begin() + (b + 1) * emb_dim);
        l2_normalize(emb);
        batch_embeddings.push_back(std::move(emb));
    }
    
    return batch_embeddings;
}

void TextEncoderTRT::l2_normalize(std::vector<float>& vec) {
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
