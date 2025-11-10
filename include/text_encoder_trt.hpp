// text_encoder_trt.hpp - Text encoder header (declarations only)
#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <memory>
#include "logger.hpp"

class TextEncoderTRT {
public:
    TextEncoderTRT();
    ~TextEncoderTRT();
    
    // Build TensorRT engine from ONNX
    bool build_engine_from_onnx(const std::string& onnx_path, const std::string& engine_path);
    
    // Initialize: load or build engine
    bool init(const std::string& onnx_path, const std::string& engine_path);
    
    // Encode single text: input token IDs, output normalized feature vector
    std::vector<float> encode(const std::vector<int64_t>& input_ids);
    
    // Encode batch texts
    std::vector<std::vector<float>> encode_batch(const std::vector<std::vector<int64_t>>& input_ids_batch);
    
private:
    void l2_normalize(std::vector<float>& vec);
    
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    cudaStream_t stream_;
    
    const int max_seq_len_ = 77;
    std::string input_name_ = "input_ids";
    std::string output_name_ = "text_emb";
};
