// image_preprocess.hpp - Image preprocessing header (declarations only)
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Configuration structure
struct ImageConfig {
    // Target size: can be square (224) or height/width (H,W)
    int out_h = 224, out_w = 224;           // Equivalent to Python's image_size
    std::string resize_mode = "shortest";   // "shortest" | "longest" | "squash"
    std::string interpolation = "bicubic";  // "bicubic" | "bilinear"
    int fill_color = 0;                     // Color for padding (longest branch CenterCropOrPad)

    // mean/std: S0/S2/B = (0,0,0)/(1,1,1), S3/S4/L-14 = CLIP mean/std
    float mean[3] = {0.481f, 0.457f, 0.406f};
    float stdv[3] = {0.268f, 0.261f, 0.275f};
};

// Image preprocessor class
class ImagePreprocessor {
public:
    ImageConfig config;
    
    // Initialize configuration
    void init(const ImageConfig& cfg);
    
    // Main preprocessing function: BGR image -> NCHW float array
    std::vector<float> preprocess(const cv::Mat& bgr, int& out_h, int& out_w);

private:
    // Get OpenCV interpolation method
    int get_interp() const;

    // Keep aspect ratio: scale with "short side=target" or "long side=target" (no crop)
    cv::Mat resize_keep_ratio(const cv::Mat& rgb, int target_h, int target_w, const std::string& mode);

    // Center crop to (H,W)
    cv::Mat center_crop(const cv::Mat& img, int H, int W);

    // Center crop or pad (pad with constant if insufficient)
    cv::Mat center_crop_or_pad(const cv::Mat& img, int H, int W, int fill = 0);

    // NHWC(RGB float [0..1]) -> NCHW + normalize
    void to_chw_normalize(const cv::Mat& rgb_float01, std::vector<float>& out);
};
