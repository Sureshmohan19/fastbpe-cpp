import time
import sys
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_hf_rust(filename, vocab_size):
    # This calls into the highly optimized Rust 'tokenizers' crate
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2)
    
    # HuggingFace handles file reading internally in Rust
    tokenizer.train([filename], trainer)
    tokenizer.save("hf_rust.json")

def train_pure_python_naive(filename, vocab_size):
    # Extremely naive implementation (approximate Karpathy minbpe)
    # Just to show how slow Python is
    with open(filename, 'rb') as f:
        text = f.read()
    
    # Integers 0-255
    ids = list(text)
    vocab = {idx: bytes([idx]) for idx in range(256)}
    merges = {}
    
    for i in range(256, vocab_size):
        stats = {}
        # Python loop overhead is massive here
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        
        if not stats: break
        pair = max(stats, key=stats.get)
        idx = i
        
        # Merge
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids

if __name__ == "__main__":
    mode = sys.argv[1]
    input_file = sys.argv[2]
    vocab_size = int(sys.argv[3])
    
    if mode == "rust":
        train_hf_rust(input_file, vocab_size)
    elif mode == "python":
        # Python is too slow for 5000 merges in a benchmark, limiting strictly
        print("Python is too slow for full bench, doing simplified run")
        train_pure_python_naive(input_file, 300)