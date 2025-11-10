// text_preprocess.cpp - Text preprocessing implementation
#include "text_preprocess.hpp"

void TextPreprocessor::init(const std::string& vocab_path, const std::string& merges_path) {
    tokenizer.load_vocab_json(vocab_path);
    tokenizer.load_merges_txt(merges_path);
}

std::vector<int> TextPreprocessor::encode(const std::string& text) {
    return tokenizer.encode(text);
}

std::vector<std::vector<int>> TextPreprocessor::encode_batch(const std::vector<std::string>& texts) {
    std::vector<std::vector<int>> results;
    results.reserve(texts.size());
    for (const auto& text : texts) {
        results.push_back(tokenizer.encode(text));
    }
    return results;
}

int TextPreprocessor::get_context_length() const {
    return tokenizer.ctx_len;
}
