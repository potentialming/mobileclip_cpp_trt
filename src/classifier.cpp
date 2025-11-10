// classifier.cpp - Classifier implementation
#include "classifier.hpp"
#include "image_preprocess.hpp"
#include "text_preprocess.hpp"
#include "image_encoder_trt.hpp"
#include "text_encoder_trt.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include <cmath>

Classifier::Classifier() {}

Classifier::~Classifier() {}

bool Classifier::init(const std::string& model_root,
                      const std::string& image_encoder_onnx,
                      const std::string& text_encoder_onnx,
                      const std::string& vocab_json,
                      const std::string& merges_txt) {
    try {
        // Build full paths
        std::string img_encoder_path = model_root + "/" + image_encoder_onnx;
        std::string txt_encoder_path = model_root + "/" + text_encoder_onnx;
        std::string vocab_path = model_root + "/" + vocab_json;
        std::string merges_path = model_root + "/" + merges_txt;
        
        std::string img_engine = img_encoder_path + ".engine";
        std::string txt_engine = txt_encoder_path + ".engine";
        
        // Initialize image encoder
        std::cout << "Initializing image encoder..." << std::endl;
        image_encoder_ = std::make_unique<ImageEncoderTRT>();
        if (!image_encoder_->init(img_encoder_path, img_engine)) {
            std::cerr << "Failed to initialize image encoder" << std::endl;
            return false;
        }
        
        // Initialize text encoder
        std::cout << "Initializing text encoder..." << std::endl;
        text_encoder_ = std::make_unique<TextEncoderTRT>();
        if (!text_encoder_->init(txt_encoder_path, txt_engine)) {
            std::cerr << "Failed to initialize text encoder" << std::endl;
            return false;
        }
        
        // Initialize image preprocessor
        ImageConfig img_config;
        img_config.out_h = img_config.out_w = 224;
        img_config.resize_mode = "shortest";
        img_config.interpolation = "bicubic";
        img_config.mean[0] = img_config.mean[1] = img_config.mean[2] = 0.0f;
        img_config.stdv[0] = img_config.stdv[1] = img_config.stdv[2] = 1.0f;
        
        image_preprocessor_ = std::make_unique<ImagePreprocessor>();
        image_preprocessor_->init(img_config);
        
        // Initialize text preprocessor
        std::cout << "Initializing tokenizer..." << std::endl;
        text_preprocessor_ = std::make_unique<TextPreprocessor>();
        text_preprocessor_->init(vocab_path, merges_path);
        
        std::cout << "Classifier initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> Classifier::encode_image(const std::string& image_path) {
    // Load image
    cv::Mat bgr = cv::imread(image_path);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    
    // Preprocess
    int h, w;
    std::vector<float> nchw = image_preprocessor_->preprocess(bgr, h, w);
    
    // Encode
    return image_encoder_->encode(nchw, 1, 3, h, w);
}

std::vector<std::vector<float>> Classifier::encode_texts(const std::vector<std::string>& texts) {
    // Tokenize all texts
    std::vector<std::vector<int>> token_ids = text_preprocessor_->encode_batch(texts);
    
    // Encode each text separately
    std::vector<std::vector<float>> text_embs;
    text_embs.reserve(texts.size());
    
    for (const auto& ids : token_ids) {
        // Convert int to int64_t
        std::vector<int64_t> ids_i64(ids.begin(), ids.end());
        // Encode single text
        std::vector<float> emb = text_encoder_->encode(ids_i64);
        text_embs.push_back(emb);
    }
    
    return text_embs;
}

float Classifier::cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

std::vector<float> Classifier::compute_similarity(const std::string& image_path,
                                                  const std::vector<std::string>& text_labels) {
    // Encode image
    std::vector<float> image_emb = encode_image(image_path);
    
    // Encode texts
    std::vector<std::vector<float>> text_embs = encode_texts(text_labels);
    
    // Compute similarity matrix
    std::vector<float> similarities;
    similarities.reserve(text_labels.size());
    
    for (size_t i = 0; i < text_labels.size(); ++i) {
        float score = cosine_similarity(image_emb, text_embs[i]);
        similarities.push_back(score);
    }
    
    return similarities;
}
