#include <queue>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>

const uint32_t BPE_MAGIC = 0x42504521;      // "BPE! in little endian format"
const uint32_t BPE_VERSION = 1;

inline uint64_t pack(uint32_t a, uint32_t b) {
    return (uint64_t(a) << 32) | b;
}

inline std::pair<uint32_t, uint32_t> unpack(uint64_t key) {
    return {uint32_t(key >> 32), uint32_t(key & 0xFFFFFFFF)};
}

// Memory Pool for Inverted Index
// Stores all pair positions in a single contiguous array. 
// Each pair keeps a linked list of positions using indices into this pool.
struct IndexPool {
    struct Node {
        int32_t pos;                                    // Position in the token stream where the pair occurs
        int32_t next;                                   // Index of the next node in this pool (-1 = end)
    };

    std::vector<Node> pool;

    IndexPool(size_t reserve_size) {
        pool.reserve(reserve_size);                     // Pre-reserve to avoid reallocations (important for performance)
    }

    inline void push(int32_t& head, int32_t pos) {
        pool.push_back({pos, head});                    // O(1) insertion: prepend a new position to the linked list
        head = static_cast<int32_t>(pool.size() - 1);   // pointed to by 'head'
    }
};

// Cache-Friendly Linear Probing Map
// Maps a packed token pair (uint64_t) to:
//   - current frequency count
//   - head of the inverted index list in IndexPool
class FastPairMap {
public:
    struct Entry {
        uint64_t key;                           // Packed (a, b) pair, UINT64_MAX = empty
        uint32_t count;                         // Current frequency of this pair
        int32_t head;                           // Head of linked list in IndexPool
    };

    std::vector<Entry> table;
    uint32_t mask;

    FastPairMap(size_t size_pow2) {
        table.resize(size_pow2, {UINT64_MAX, 0, -1});
        mask = static_cast<uint32_t>(size_pow2 - 1);
    }

    // Lookup or insertion slot for a key.
    // Returns either:
    //   - pointer to existing entry
    //   - pointer to an empty slot where the key can be inserted
    inline Entry* get(uint64_t key) {
        uint32_t idx = (key * 0x9E3779B97F4A7C15ULL) & mask;
        while (true) {
            if (table[idx].key == key) return &table[idx];
            if (table[idx].key == UINT64_MAX) return &table[idx];
            idx = (idx + 1) & mask;
        }
    }
};

class BPETokenizer {
public:
    struct MergeRule {
        uint32_t a, b, new_id;
    };

    std::vector<std::string> vocab;
    std::vector<MergeRule> merges;
    
    // For inference (Encode) - lazy initialized
    FastPairMap inference_map = FastPairMap(16);
    
    BPETokenizer() {
        vocab.reserve(10000);
        for (int i = 0; i < 256; i++) {
            vocab.push_back(std::string(1, (char)i));
        }
    }

    // MANUAL LEXER (no regex)
    // This version is intentionally simple, fast, and byte-oriented.
    // TODO: Replace this with a proper byte-level FSM that more closely matches
    void lexical_split(const std::string& text,
                    std::vector<uint32_t>& val,
                    std::vector<int32_t>& next) {

        const size_t n = text.size();
        size_t i = 0;

        while (i < n) {
            const size_t start = i;
            const unsigned char c = static_cast<unsigned char>(text[i]);

            // 1. Whitespace segment
            if (std::isspace(c)) {
                while (i < n && std::isspace(static_cast<unsigned char>(text[i]))) {
                    i++;
                }
            }
            // 2. Alphabetic segment (ASCII only)
            else if (std::isalpha(c)) {
                while (i < n && std::isalpha(static_cast<unsigned char>(text[i]))) {
                    i++;
                }
            }
            // 3. Numeric segment
            else if (std::isdigit(c)) {
                while (i < n && std::isdigit(static_cast<unsigned char>(text[i]))) {
                    i++;
                }
            }
            // 4. Punctuation / other bytes (single-byte segment)
            else {
                i++;
            }

            // Emit bytes for this segment
            const size_t segment_begin = val.size();
            for (size_t k = start; k < i; k++) {
                val.push_back(static_cast<unsigned char>(text[k]));
                next.push_back(-1);  // temporarily mark as end
            }

            // Link tokens within the segment
            const size_t segment_end = val.size();
            for (size_t p = segment_begin; p + 1 < segment_end; p++) {
                next[p] = static_cast<int32_t>(p + 1);
            }
            // next[segment_end - 1] stays -1 (segment boundary)
        }
    }


    // train BPE tokenizer on full in-memory text (no streaming yet)
    void train(const std::string& text, uint32_t target_vocab, uint32_t min_freq) {

        if (target_vocab <= 256) return;                        // No merges possible below byte-level vocab
        
        size_t est_tokens = text.size();                        // one token per byte
        std::vector<uint32_t> val;  val.reserve(est_tokens);    // Token values (byte IDs / merged IDs)
        std::vector<int32_t>  next; next.reserve(est_tokens);   // Next pointer (linked list)
        std::vector<int32_t>  prev;                             // Prev pointer (built after lexing)

        lexical_split(text, val, next);
    
        size_t n = val.size();
        prev.resize(n, -1);                                     // -1 means no previous token (segment start)
        for (size_t i = 0; i < n; i++) {
            if (next[i] != -1 && next[i] < (int32_t)n) {
                prev[next[i]] = i;                              // Record backward link
            }
        }

        uint32_t map_size = 1;                                      // Choose hash table size as a power of two for fast masking, 
        while (map_size < target_vocab * 4) map_size <<= 1;         // oversized to reduce collisions during training

        FastPairMap stats(map_size);                                // Hash map: (token_a, token_b) -> {frequency, list of positions}
        IndexPool index_pool(n / 2);                                // Memory pool storing all pair positions as intrusive linked lists
        std::priority_queue<std::pair<uint32_t, uint64_t>> queue;   // Max-heap: (pair_count, pair_key) to always pick the most frequent pair

        for (size_t i = 0; i < n; i++) {

            if (next[i] == -1) continue;                            // Skip segment boundaries
            uint64_t key = pack(val[i], val[next[i]]);              // Encode adjacent token pair into a single 64-bit key

            auto* entry = stats.get(key);
            if (entry->key == UINT64_MAX) {
                entry->key = key;
                entry->count = 0;
                entry->head = -1;
            }

            entry->count++;
            index_pool.push(entry->head, i);
        }

        size_t unique_pairs = 0;                        
                                                        
        for (const auto& entry : stats.table) {
            if (entry.key != UINT64_MAX && entry.count >= min_freq) {
                queue.push({entry.count, entry.key});                   // Populate the priority queue with all frequent pairs,
                unique_pairs++;                                         // so we can always select the most frequent pair to merge next.
            }
        }
        
        uint32_t current_vocab = 256;
        uint32_t merges_to_do = target_vocab - 256;
        uint32_t skipped = 0;
        
        while (current_vocab < target_vocab) {

            if (queue.empty()) break;                                   // No merge candidates left
            
            auto top = queue.top();
            queue.pop();

            uint32_t count = top.first;
            uint64_t pair  = top.second;

            auto* entry = stats.get(pair);
            if (entry->key == UINT64_MAX) {
                skipped++;
                continue;
            }
            if (entry->count != count) {
                skipped++;
                continue;
            }
            if (entry->count < min_freq) {
                break;
            }

            uint32_t merges_done = current_vocab - 256;
            uint32_t new_token = current_vocab++;
            auto parts = unpack(pair);
            
            vocab.push_back(vocab[parts.first] + vocab[parts.second]);      // Record merge rule and token string
            merges.push_back({parts.first, parts.second, new_token});

            int32_t saved_head = entry->head;                               // Save inverted index head BEFORE invalidating

            entry->key = UINT64_MAX;
            entry->count = 0;
            entry->head = -1;


            // Collect all positions where this pair occurs (snapshot)
            std::vector<int32_t> positions;
            int32_t walk = saved_head;

            while (walk != -1 && walk < (int32_t)index_pool.pool.size()) {
                positions.push_back(index_pool.pool[walk].pos);
                walk = index_pool.pool[walk].next;
            }

            std::sort(positions.begin(), positions.end());
            positions.erase(std::unique(positions.begin(), positions.end()), positions.end());

        #ifndef NDEBUG
            for (int32_t pos : positions) { assert(pos >= 0 && pos < (int32_t)val.size()); }
        #endif

            
            for (int32_t pos : positions) {

                if (pos < 0 || pos >= (int32_t)val.size()) continue;
                if (val[pos] != parts.first) continue;

                int32_t next_pos = next[pos];
                if (next_pos < 0 || next_pos >= (int32_t)val.size()) continue;
                if (val[next_pos] != parts.second) continue;

                int32_t p  = prev[pos];
                int32_t nn = next[next_pos];

                // stale-position guards 
                // If the links are no longer consistent, this position is stale.
                // Skip it safely.
                if (p != -1 && next[p] != pos) continue;
                if (nn != -1 && prev[nn] != next_pos) continue;

            #ifndef NDEBUG
                if (p  != -1) assert(next[p] == pos);
                if (nn != -1) assert(prev[nn] == next_pos);
            #endif

                // Decrement old neighboring pairs
                if (p >= 0) {
                    auto* e = stats.get(pack(val[p], parts.first));
                    if (e->key != UINT64_MAX && e->count > 0) {
                        e->count--;
            #ifndef NDEBUG
                        assert(e->count >= 0);
            #endif
                    }
                }

                if (nn >= 0) {
                    auto* e = stats.get(pack(parts.second, val[nn]));
                    if (e->key != UINT64_MAX && e->count > 0) {
                        e->count--;
            #ifndef NDEBUG
                        assert(e->count >= 0);
            #endif
                    }
                }

                val[pos] = new_token;
                next[pos] = nn;
                if (nn >= 0) {
                    prev[nn] = pos;
                }

            #ifndef NDEBUG
                // Ensure removed token is no longer reachable
                assert(next[pos] != pos);
            #endif

                // Increment new neighboring pairs
                if (p >= 0) {
                    uint64_t key = pack(val[p], new_token);
                    auto* e = stats.get(key);
                    if (e->key == UINT64_MAX) {
                        e->key = key;
                        e->count = 0;
                        e->head = -1;
                    }
                    e->count++;
                    index_pool.push(e->head, p);

                    if (e->count >= min_freq) {
                        queue.push({e->count, key});
                    }
                }

                if (nn >= 0) {
                    uint64_t key = pack(new_token, val[nn]);
                    auto* e = stats.get(key);
                    if (e->key == UINT64_MAX) {
                        e->key = key;
                        e->count = 0;
                        e->head = -1;
                    }
                    e->count++;
                    index_pool.push(e->head, pos);

                    if (e->count >= min_freq) {
                        queue.push({e->count, key});
                    }
                }
            }
        }
    }

    // Binary layout (little-endian, same-arch) for saving tokenizer to disk in binary format:
    //   [magic:u32][version:u32]
    //   [vocab_size:u32][merge_count:u32]
    //   [MergeRule x merge_count]
    //   [ [token_len:u32][token_bytes] x vocab_size ]
    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open file for writing");
        }

        out.write(reinterpret_cast<const char*>(&BPE_MAGIC), sizeof(BPE_MAGIC));
        out.write(reinterpret_cast<const char*>(&BPE_VERSION), sizeof(BPE_VERSION));

        uint32_t vocab_size  = static_cast<uint32_t>(vocab.size());
        uint32_t merge_count = static_cast<uint32_t>(merges.size());

        out.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        out.write(reinterpret_cast<const char*>(&merge_count), sizeof(merge_count));

        
        for (const auto& m : merges) {
            out.write(reinterpret_cast<const char*>(&m), sizeof(m));        // Each MergeRule is written as raw bytes (POD, fixed-size).
        }
        
        for (const auto& token : vocab) {
            uint32_t len = static_cast<uint32_t>(token.size());
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(token.data(), len);
        }

        if (!out) {
            throw std::runtime_error("Error occurred while writing tokenizer file");
        }
    }

    // Load tokenizer from a binary file previously written by `save()`.
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("File not found");
        }

        uint32_t magic, version;
        in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        in.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != BPE_MAGIC) {
            throw std::runtime_error("Invalid file format (bad magic number)");
        }
        if (version != BPE_VERSION) {
            throw std::runtime_error("Unsupported file version");
        }

        uint32_t vs, ms;
        in.read(reinterpret_cast<char*>(&vs), sizeof(vs));
        in.read(reinterpret_cast<char*>(&ms), sizeof(ms));

        if (vs > 1'000'000 || ms > 1'000'000) {
            throw std::runtime_error("Suspicious vocab or merge count");
        }

        merges.resize(ms);
        in.read(reinterpret_cast<char*>(merges.data()),
                ms * sizeof(MergeRule));

        vocab.clear();
        vocab.reserve(vs);

        for (uint32_t i = 0; i < vs; i++) {
            uint32_t len;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));

            if (len > 1000) {
                throw std::runtime_error("Suspicious token length");
            }

            std::string token(len, '\0');
            in.read(&token[0], len);
            vocab.push_back(std::move(token));
        }

        if (!in) {
            throw std::runtime_error("File read error");
        }

        // Build inference structures - Precompute fast lookup tables for encoding.
        build_inference_map();
    }
    
    // Build fast lookup table for inference from learned merge rules.
    // Maps (a, b) token pairs -> merge rank (stored in Entry::head). This allows O(1) average-time lookup during encoding.
    void build_inference_map() {
        uint32_t map_size = 1;
        while (map_size < merges.size() * 2) {
            map_size <<= 1;
        }
        
        inference_map.table.assign(map_size, {UINT64_MAX, 0, -1});      // Initialize hash table: UINT64_MAX marks empty slots.
        inference_map.mask = map_size - 1;

        for (size_t i = 0; i < merges.size(); i++) {                    // Insert all merge rules into the hash table.
            uint64_t key = pack(merges[i].a, merges[i].b);              // The index 'i' is the merge rank (lower = higher priority).
            
            uint32_t idx =
                (key * 0x9E3779B97F4A7C15ULL) & inference_map.mask;     // Multiplicative hash + linear probing

            while (true) {
                if (inference_map.table[idx].key == UINT64_MAX) {
                    inference_map.table[idx].key = key;
                    inference_map.table[idx].head = static_cast<int32_t>(i);
                    break;
                }
                idx = (idx + 1) & inference_map.mask;
            }
        }
    }


    // Encode a single contiguous token segment using learned BPE merge rules.
    // Repeatedly applies the highest-priority (lowest-rank) merge until no more apply.
    std::vector<uint32_t> byte_pair_encode_piece(const std::vector<uint32_t>& piece) {

        if (piece.size() < 2) return piece;

        std::vector<uint32_t> work = piece;

        while (work.size() >= 2) {
            int32_t best_rank = INT32_MAX;
            size_t best_i = 0;

            for (size_t i = 0; i + 1 < work.size(); i++) {
                uint64_t key = pack(work[i], work[i + 1]);
                auto* e = inference_map.get(key);

                if (e->key != UINT64_MAX) {
                    int32_t rank = e->head;  
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_i = i;
                    }
                }
            }

            if (best_rank == INT32_MAX) break;

            uint32_t new_token = merges[best_rank].new_id;
            work[best_i] = new_token;
            work.erase(work.begin() + best_i + 1);
        }

        return work;
    }


    // Encode input text into BPE token IDs using trained merge rules.
    std::vector<uint32_t> encode(const std::string& text) {

        // Lazily build inference lookup table if not already initialized.
        // This is needed after load() or train().
        if (inference_map.table[0].key == UINT64_MAX && !merges.empty()) {
            build_inference_map();
        }

        std::vector<uint32_t> ids;
        std::vector<int32_t> next_arr;
        lexical_split(text, ids, next_arr);

        std::vector<uint32_t> result;
        result.reserve(ids.size());                         // Upper bound: no more tokens than bytes

        std::vector<uint32_t> segment;
        segment.reserve(32);                                // Typical segments are small; avoids frequent reallocs

        for (size_t i = 0; i < ids.size(); i++) {
            segment.push_back(ids[i]);

            if (next_arr[i] == -1) {                    
                auto encoded = byte_pair_encode_piece(segment);
                result.insert(result.end(), encoded.begin(), encoded.end());
                segment.clear();  
            }
        }

        return result;
    }

    // Decode token IDs back into the original byte sequence.
    std::string decode(const std::vector<uint32_t>& ids) {
        std::string s;
        s.reserve(ids.size()); 

        for (uint32_t id : ids) {

    #ifndef NDEBUG
            assert(id < vocab.size());
    #endif

            if (id < vocab.size()) {
                s += vocab[id];
            }
        }
        return s;
    }

};

// NOTE: Reads entire file into memory; fine for initial implementation.
// Streaming I/O will be handled at a higher layer (e.g., Python) later.
std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open file");
    size_t sz = in.tellg();
    std::string s(sz, '\0');
    in.seekg(0);
    in.read(&s[0], sz);
    return s;
}

// main function
int main(int argc, char** argv) {
    if (argc < 2) return 1;         // Require at least a command name

    std::string cmd = argv[1];      // Command: train | encode | decode
    BPETokenizer tok;
    
    if (cmd == "train") {
        auto text = read_file(argv[2]);                             // Read training corpus
        uint32_t vs = std::stoi(argv[4]);                           // Vocabulary size
        uint32_t min_freq = (argc > 5) ? std::stoi(argv[5]) : 2;    // Min merge frequency
        tok.train(text, vs, min_freq);                              // Learn BPE merges
        tok.save(argv[3]);                                          // Save tokenizer model
        std::cout << "Done.\n";
    }
    else if (cmd == "encode") {
        tok.load(argv[2]);                                          // Load trained tokenizer
        auto ids = tok.encode(argv[3]);                             // Encode text into token IDs
        for(auto id : ids) std::cout << id << " ";
        std::cout << "\n";
    }
    else if (cmd == "decode") {
        tok.load(argv[2]);                                          // Load trained tokenizer
        std::vector<uint32_t> ids;
        for(int i=3; i<argc; i++)                                   // Parse token IDs from CLI
            ids.push_back(std::stoi(argv[i]));
        std::cout << tok.decode(ids) << "\n";                       // Decode IDs back to text
    }
    return 0;
}