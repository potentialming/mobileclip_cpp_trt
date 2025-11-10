// image_encoder_trt.hpp - Image encoder header (declarations only)
#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <memory>
#include "logger.hpp"

class ImageEncoderTRT {
public:
    ImageEncoderTRT();
    ~ImageEncoderTRT();
    
    // Build TensorRT engine from ONNX
    bool build_engine_from_onnx(const std::string& onnx_path, const std::string& engine_path);
    
    // Initialize: load or build engine
    bool init(const std::string& onnx_path, const std::string& engine_path);
    
    // Encode image: input NCHW float array, output normalized feature vector
    std::vector<float> encode(const std::vector<float>& nchw_data, int batch, int channels, int height, int width);
    
private:
    void l2_normalize(std::vector<float>& vec);
    
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    cudaStream_t stream_;
    
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    
    std::vector<float> host_output_;
    
    std::string input_name_ = "pixel_values";
    std::string output_name_ = "image_emb";
};
