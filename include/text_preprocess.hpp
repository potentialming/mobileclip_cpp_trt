// text_preprocess.hpp - Text preprocessing header (declarations only)
#pragma once
#include "tokenizer.hpp"
#include <vector>
#include <string>

// Text preprocessor class
class TextPreprocessor {
public:
    BPE tokenizer;
    
    // Initialize tokenizer (requires vocab.json and merges.txt)
    void init(const std::string& vocab_path, const std::string& merges_path);
    
    // Encode single text
    std::vector<int> encode(const std::string& text);
    
    // Encode batch of texts
    std::vector<std::vector<int>> encode_batch(const std::vector<std::string>& texts);
    
    // Get context length
    int get_context_length() const;
};
