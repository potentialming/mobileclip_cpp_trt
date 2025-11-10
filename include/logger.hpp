// logger.hpp - TensorRT Logger
#pragma once
#include <NvInfer.h>
#include <iostream>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};
