// tokenizer.hpp - CLIP text tokenizer header (declarations only)
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

// bytes_to_unicode: GPT-2/CLIP classic mapping
std::unordered_map<unsigned char, std::string> bytes_to_unicode();

// BPE tokenizer struct
struct BPE {
    std::unordered_map<std::string, int> encoder;        // token -> id
    std::unordered_map<std::string, int> bpe_ranks;      // "A B" -> rank
    std::unordered_map<std::string, std::string> cache;  // BPE cache
    int bos_id = 49406, eos_id = 49407;                  // CLIP special tokens
    int ctx_len = 77;

    static std::string trim(const std::string& s);
    
    // Load vocab.json (token->id mapping)
    void load_vocab_json(const std::string& path);
    
    // Load merges.txt (merge pairs->rank)
    void load_merges_txt(const std::string& path);
    
    // Get best pair from adjacent tokens with minimum rank
    static std::pair<int, std::string> best_pair(
        const std::vector<std::string>& toks,
        const std::unordered_map<std::string, int>& ranks);
    
    // Apply BPE merging to a token
    std::string bpe(const std::string& token);
    
    // Encode text to token IDs (with BOS/EOS, fixed length 77)
    std::vector<int> encode(const std::string& text);
};
