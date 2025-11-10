// classifier.hpp - MobileClip classifier class
#pragma once
#include <vector>
#include <string>
#include <memory>

// Forward declarations
class ImageEncoderTRT;
class TextEncoderTRT;
class ImagePreprocessor;
class TextPreprocessor;
struct ImageConfig;

class Classifier {
public:
    Classifier();
    ~Classifier();
    
    // Initialize classifier with model paths
    bool init(const std::string& model_root,
              const std::string& image_encoder_onnx,
              const std::string& text_encoder_onnx,
              const std::string& vocab_json,
              const std::string& merges_txt);
    
    // Compute similarity matrix between image and text labels
    // Returns: similarity_matrix[i] = similarity between image and text_labels[i]
    std::vector<float> compute_similarity(const std::string& image_path,
                                          const std::vector<std::string>& text_labels);

private:
    std::unique_ptr<ImageEncoderTRT> image_encoder_;
    std::unique_ptr<TextEncoderTRT> text_encoder_;
    std::unique_ptr<ImagePreprocessor> image_preprocessor_;
    std::unique_ptr<TextPreprocessor> text_preprocessor_;
    
    std::vector<float> encode_image(const std::string& image_path);
    std::vector<std::vector<float>> encode_texts(const std::vector<std::string>& texts);
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
};
