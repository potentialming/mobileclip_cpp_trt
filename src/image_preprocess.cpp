// image_preprocess.cpp - Image preprocessing implementation
#include "image_preprocess.hpp"
#include <algorithm>

void ImagePreprocessor::init(const ImageConfig& cfg) {
    config = cfg;
}

std::vector<float> ImagePreprocessor::preprocess(const cv::Mat& bgr, int& out_h, int& out_w) {
    // 1) BGR->RGB
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    // 2) Resize (by shortest/longest/squash)
    cv::Mat res;
    if (config.resize_mode == "squash") {
        // Directly stretch to (H,W), not maintaining aspect ratio
        cv::resize(rgb, res, cv::Size(config.out_w, config.out_h), 0, 0, get_interp());
    } else if (config.resize_mode == "longest") {
        // First align long side to target long side, then CenterCropOrPad to (H,W)
        res = resize_keep_ratio(rgb, config.out_h, config.out_w, "longest");
        res = center_crop_or_pad(res, config.out_h, config.out_w, config.fill_color);
    } else {
        // shortest (most common): first align short side to target short side, then CenterCrop to (H,W)
        res = resize_keep_ratio(rgb, config.out_h, config.out_w, "shortest");
        res = center_crop(res, config.out_h, config.out_w);
    }

    // 3) Convert to float32 [0,1]
    cv::Mat f;
    res.convertTo(f, CV_32F, 1.0/255.0);

    // 4) ToTensor + Normalize (CLIP mean/std or 0/1)
    std::vector<float> nchw;
    to_chw_normalize(f, nchw);

    out_h = config.out_h;
    out_w = config.out_w;
    return nchw;
}

int ImagePreprocessor::get_interp() const {
    return (config.interpolation == "bilinear") ? cv::INTER_LINEAR : cv::INTER_CUBIC;
}

cv::Mat ImagePreprocessor::resize_keep_ratio(const cv::Mat& rgb, int target_h, int target_w, const std::string& mode) {
    int ih = rgb.rows, iw = rgb.cols;
    double scale_short = std::min(target_h, target_w) / (double)std::min(ih, iw);
    double scale_long = std::max(target_h, target_w) / (double)std::max(ih, iw);
    double s = (mode == "shortest") ? scale_short : scale_long;
    int nh = std::max(1, (int)std::round(ih * s));
    int nw = std::max(1, (int)std::round(iw * s));
    cv::Mat out;
    cv::resize(rgb, out, cv::Size(nw, nh), 0, 0, get_interp());
    return out;
}

cv::Mat ImagePreprocessor::center_crop(const cv::Mat& img, int H, int W) {
    int y = std::max(0, (img.rows - H) / 2);
    int x = std::max(0, (img.cols - W) / 2);
    int h = std::min(H, img.rows), w = std::min(W, img.cols);
    return img(cv::Rect(x, y, w, h)).clone();
}

cv::Mat ImagePreprocessor::center_crop_or_pad(const cv::Mat& img, int H, int W, int fill) {
    cv::Mat canvas(H, W, img.type(), cv::Scalar(fill, fill, fill));
    int h = std::min(H, img.rows), w = std::min(W, img.cols);
    // Crop
    int y = std::max(0, (img.rows - H) / 2);
    int x = std::max(0, (img.cols - W) / 2);
    cv::Mat crop = img(cv::Rect(x, y, w, h));
    // Paste to center
    int dy = (H - h) / 2, dx = (W - w) / 2;
    crop.copyTo(canvas(cv::Rect(dx, dy, w, h)));
    return canvas;
}

void ImagePreprocessor::to_chw_normalize(const cv::Mat& rgb_float01, std::vector<float>& out) {
    int H = rgb_float01.rows, W = rgb_float01.cols;
    out.resize(3 * H * W);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < H; ++y) {
            const cv::Vec3f* row = rgb_float01.ptr<cv::Vec3f>(y);
            for (int x = 0; x < W; ++x) {
                float v = row[x][c];
                v = (v - config.mean[c]) / config.stdv[c];
                out[idx++] = v;
            }
        }
    }
}
