// tokenizer.cpp - CLIP text tokenizer implementation
#include "tokenizer.hpp"
#include <algorithm>
#include <regex>
#include <climits>
#include <sstream>
#include <stdexcept>

// bytes_to_unicode: GPT-2/CLIP classic mapping
std::unordered_map<unsigned char, std::string> bytes_to_unicode() {
    std::vector<int> bs;
    for (int c='!'; c<='~'; ++c) bs.push_back(c);
    for (int c=0xA1; c<=0xAC; ++c) bs.push_back(c);
    for (int c=0xAE; c<=0xFF; ++c) bs.push_back(c);
    std::vector<int> cs = bs;
    int n = 0;
    for (int b=0; b<256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    std::unordered_map<unsigned char, std::string> map;
    for (size_t i=0; i<bs.size(); ++i) {
        map[(unsigned char)bs[i]] = std::string(1, (char)cs[i]);
    }
    return map;
}

std::string BPE::trim(const std::string& s) {
    auto a = s.find_first_not_of(" \t\r\n");
    auto b = s.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

void BPE::load_vocab_json(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open vocab.json: " + path);
    
    std::string content((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    
    // Simple parsing: find "token": id format
    std::string pattern_str = "\"([^\"]+)\"\\s*:\\s*(\\d+)";
    std::regex pattern(pattern_str);
    std::smatch match;
    std::string::const_iterator search_start(content.cbegin());
    
    while (std::regex_search(search_start, content.cend(), match, pattern)) {
        std::string token = match[1].str();
        int id = std::stoi(match[2].str());
        encoder[token] = id;
        search_start = match.suffix().first;
    }
    
    if (encoder.empty()) {
        throw std::runtime_error("Failed to parse vocab.json or vocab is empty");
    }
}

void BPE::load_merges_txt(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Cannot open merges.txt: " + path);
    
    std::string line;
    int rank = 0;
    // First line is usually "#version: x.y"
    std::getline(fin, line);
    while (std::getline(fin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        bpe_ranks[line] = rank++;
    }
}

std::pair<int, std::string> BPE::best_pair(
    const std::vector<std::string>& toks,
    const std::unordered_map<std::string, int>& ranks) {
    int best = INT_MAX;
    std::string best_pair;
    for (size_t i = 0; i + 1 < toks.size(); ++i) {
        std::string key = toks[i] + " " + toks[i + 1];
        auto it = ranks.find(key);
        if (it != ranks.end() && it->second < best) {
            best = it->second;
            best_pair = key;
        }
    }
    return {best, best_pair};
}

std::string BPE::bpe(const std::string& token) {
    auto itc = cache.find(token);
    if (itc != cache.end()) return itc->second;
    
    // Split into character sequence, add </w> to last character
    std::vector<std::string> word;
    for (size_t i = 0; i < token.size(); ++i) {
        if (i == token.size() - 1) {
            // Add </w> marker to last character
            word.push_back(token.substr(i, 1) + "</w>");
        } else {
            word.push_back(token.substr(i, 1));
        }
    }
    if (word.size() == 1) return cache[token] = word[0];

    while (true) {
        auto [rank, bp] = best_pair(word, bpe_ranks);
        if (rank == INT_MAX) break;
        // Merge the best pair
        std::vector<std::string> new_word;
        for (size_t i = 0; i < word.size(); ) {
            if (i + 1 < word.size() && (word[i] + " " + word[i + 1]) == bp) {
                new_word.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }
        word.swap(new_word);
        if (word.size() == 1) break;
    }
    // Restore space-separated BPE result
    std::ostringstream oss;
    for (size_t i = 0; i < word.size(); ++i) {
        if (i) oss << ' ';
        oss << word[i];
    }
    return cache[token] = oss.str();
}

std::vector<int> BPE::encode(const std::string& text) {
    static auto byte_enc = bytes_to_unicode();

    // (a) Normalize: lowercase + whitespace cleanup
    std::string t = text;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    
    // (b) Pre-tokenize: split by whitespace
    std::vector<std::string> tokens;
    {
        std::regex ws("\\S+");
        std::smatch m;
        auto it = t.cbegin();
        while (std::regex_search(it, t.cend(), m, ws)) {
            tokens.push_back(m[0]);
            it = m.suffix().first;
        }
    }

    // (c) Byte-level: convert each token to bytes and map to unicode
    std::vector<int> ids;
    ids.reserve(ctx_len);
    ids.push_back(bos_id);
    for (auto& tok : tokens) {
        std::string trans;
        trans.reserve(tok.size() * 2);
        for (unsigned char ch : tok) {
            trans += byte_enc[ch];
        }

        // (d) BPE merge and lookup to id
        std::istringstream iss(bpe(trans));
        std::string piece;
        while (iss >> piece) {
            auto it = encoder.find(piece);
            if (it != encoder.end()) {
                ids.push_back(it->second);
            }
        }
        if ((int)ids.size() >= ctx_len - 1) break; // Reserve for EOS
    }
    ids.push_back(eos_id);
    // Pad to 77
    while ((int)ids.size() < ctx_len) {
        ids.push_back(0);
    }
    if ((int)ids.size() > ctx_len) {
        ids.resize(ctx_len);
    }
    return ids;
}
