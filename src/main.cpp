// main.cpp - Main entry point for MobileClip classifier
#include "classifier.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Simple config structure
struct Config {
    std::string model_root;
    std::string image_encoder_onnx;
    std::string text_encoder_onnx;
    std::string vocab_json;
    std::string merges_txt;
    std::string test_image;
    std::vector<std::string> text_labels;
};

// Parse config file (simplified version)
Config parse_config(const std::string& config_path) {
    Config config;
    // Hardcoded for now - can be replaced with yaml-cpp library
    config.model_root = "models";
    config.image_encoder_onnx = "image_encoder.onnx";
    config.text_encoder_onnx = "text_encoder.onnx";
    config.vocab_json = "vocab.json";
    config.merges_txt = "merges.txt";
    config.test_image = "images/traffic_light.jpg";
    config.text_labels = {
        "the red light is on",
        "the yellow light is on",
        "the green light is on",
        "no light is on"
    };
    
    std::cout << "Config loaded from: " << config_path << std::endl;
    return config;
}

// Find the index with maximum similarity
int find_max_index(const std::vector<float>& similarities) {
    if (similarities.empty()) return -1;
    
    int max_idx = 0;
    float max_val = similarities[0];
    
    for (size_t i = 1; i < similarities.size(); ++i) {
        if (similarities[i] > max_val) {
            max_val = similarities[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " config/config.yaml" << std::endl;
        return 0;
    }
    
    std::string config_path = argv[1];
    
    std::cout << "\n=== MobileClip TensorRT Classifier ===" << std::endl;
    std::cout << "Loading configuration..." << std::endl;
    
    // Parse config
    Config config = parse_config(config_path);
    
    try {
        // Initialize classifier
        std::cout << "\n=== Initializing Classifier ===" << std::endl;
        Classifier classifier;
        
        if (!classifier.init(config.model_root,
                            config.image_encoder_onnx,
                            config.text_encoder_onnx,
                            config.vocab_json,
                            config.merges_txt)) {
            std::cerr << "Failed to initialize classifier" << std::endl;
            return 1;
        }
        
        // Run classification
        std::cout << "\n=== Running Classification ===" << std::endl;
        std::cout << "Image: " << config.test_image << std::endl;
        std::cout << "Text labels: " << config.text_labels.size() << std::endl;
        
        std::vector<float> similarities = classifier.compute_similarity(
            config.test_image, config.text_labels);
        
        // Display results
        std::cout << "\n=== Similarity Matrix ===" << std::endl;
        for (size_t i = 0; i < config.text_labels.size(); ++i) {
            std::cout << "  [" << i << "] " << config.text_labels[i] 
                      << ": " << similarities[i] << std::endl;
        }
        
        // Find and display prediction
        int max_idx = find_max_index(similarities);
        if (max_idx >= 0) {
            std::cout << "\n=== Prediction ===" << std::endl;
            std::cout << "Best match: " << config.text_labels[max_idx] << std::endl;
            std::cout << "Confidence: " << similarities[max_idx] << std::endl;
        }
        
        std::cout << "\n=== Classification Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
