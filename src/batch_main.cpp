// batch_main.cpp - Batch processing for traffic light classification
#include "classifier.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// Configuration structure
struct BatchConfig {
    std::string model_root;
    std::string image_encoder_onnx;
    std::string text_encoder_onnx;
    std::string vocab_json;
    std::string merges_txt;
    std::string dataset_path;
    std::string output_path;
    std::vector<std::string> text_labels;
    std::vector<std::string> class_names;
};

// Parse YAML config (simplified)
BatchConfig parse_config(const std::string& config_path) {
    BatchConfig config;
    
    // Hardcoded paths from traffic_light_classification.yaml
    config.model_root = "models";
    config.image_encoder_onnx = "image_encoder.onnx";
    config.text_encoder_onnx = "text_encoder.onnx";
    config.vocab_json = "vocab.json";
    config.merges_txt = "merges.txt";
    config.dataset_path = "/home/liming/datasets/cropped_traffic_light";
    config.output_path = "outputs/traffic_light";
    
    config.text_labels = {
        "the red light is on",
        "the yellow light is on",
        "the green light is on",
        "no light is on"
    };
    
    config.class_names = {
        "RED",
        "YELLOW",
        "GREEN",
        "OFF"
    };
    
    std::cout << "Config loaded from: " << config_path << std::endl;
    return config;
}

// Get all image files from dataset
std::vector<std::string> get_all_images(const std::string& dataset_path) {
    std::vector<std::string> image_files;
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"};
    std::vector<std::string> subdirs = {"red", "yellow", "green"};
    
    for (const auto& subdir : subdirs) {
        std::string subdir_path = dataset_path + "/" + subdir;
        
        if (!fs::exists(subdir_path)) {
            continue;
        }
        
        for (const auto& entry : fs::directory_iterator(subdir_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    image_files.push_back(entry.path().string());
                }
            }
        }
    }
    
    std::sort(image_files.begin(), image_files.end());
    return image_files;
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

// Extract filename without extension
std::string get_filename_without_ext(const std::string& filepath) {
    fs::path p(filepath);
    return p.stem().string();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " config/traffic_light_classification.yaml" << std::endl;
        return 0;
    }
    
    std::string config_path = argv[1];
    
    std::cout << "\n=== MobileClip Batch Traffic Light Classifier ===" << std::endl;
    std::cout << "Loading configuration..." << std::endl;
    
    // Parse config
    BatchConfig config = parse_config(config_path);
    
    try {
        // Create output directory
        fs::create_directories(config.output_path);
        std::cout << "Output directory: " << config.output_path << std::endl;
        
        // Get all images
        std::cout << "\nScanning dataset: " << config.dataset_path << std::endl;
        std::vector<std::string> all_images = get_all_images(config.dataset_path);
        std::cout << "Found " << all_images.size() << " images" << std::endl;
        
        if (all_images.empty()) {
            std::cerr << "No images found in dataset!" << std::endl;
            return 1;
        }
        
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
        
        // Process all images
        std::cout << "\n=== Processing Images ===" << std::endl;
        int success_count = 0;
        int failed_count = 0;
        
        for (size_t idx = 0; idx < all_images.size(); ++idx) {
            const std::string& image_path = all_images[idx];
            std::string filename = get_filename_without_ext(image_path);
            std::string output_file = config.output_path + "/" + filename + ".txt";
            
            // Progress indicator
            if ((idx + 1) % 100 == 0 || idx == 0) {
                std::cout << "Processing [" << (idx + 1) << "/" << all_images.size() << "] " 
                          << filename << std::endl;
            }
            
            try {
                // Classify image
                std::vector<float> similarities = classifier.compute_similarity(
                    image_path, config.text_labels);
                
                // Find best match
                int max_idx = find_max_index(similarities);
                
                if (max_idx >= 0 && max_idx < static_cast<int>(config.class_names.size())) {
                    // Save result to file
                    std::ofstream outfile(output_file);
                    if (outfile.is_open()) {
                        outfile << config.class_names[max_idx];
                        outfile.close();
                        success_count++;
                    } else {
                        std::cerr << "Failed to write: " << output_file << std::endl;
                        failed_count++;
                    }
                } else {
                    std::cerr << "Invalid classification result for: " << image_path << std::endl;
                    failed_count++;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing " << image_path << ": " << e.what() << std::endl;
                failed_count++;
            }
        }
        
        // Print summary
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Processing complete!" << std::endl;
        std::cout << "Total images: " << all_images.size() << std::endl;
        std::cout << "Successfully processed: " << success_count << std::endl;
        std::cout << "Failed: " << failed_count << std::endl;
        std::cout << "Results saved to: " << config.output_path << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
