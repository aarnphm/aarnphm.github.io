from typing import Dict, Tuple
import functools
from io import BufferedReader
import multiprocessing
import os
import regex as re
from collections import Counter, defaultdict

from tqdm import tqdm, trange
import psutil
import time
import threading

# TODOs: 
# -  Identify inefficiencies and bugs in the code
# -  How does memory usage scale with file size and number of processes?
# -  How can you speed up the code?
# -  Build monitoring tools to track memory usage, speed, and other metrics

PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKENIZER = re.compile(PRETOKENIZER_PATTERN)

def get_file_length(file_path: str) -> int:
    return os.path.getsize(file_path)

def pretokenize_chunk(start_end_indices: Tuple[int, int], special_token: str, content: str) -> Counter:
    start, end = start_end_indices
    content_chunk = content[start:end]
    if content_chunk.startswith(special_token):
        content_chunk = content_chunk[len(special_token):]
    pretokens = PRETOKENIZER.findall(content_chunk)
    counter = Counter(pretokens)
    return counter

def memory_reporter(stop_event, interval=1.0):
    """Thread function to print memory usage every interval seconds."""
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        mem = process.memory_info().rss / (1024 * 1024)
        print(f"\r[Memory] RSS: {mem:.2f} MB", end="", flush=True)
        time.sleep(interval)
    # Print final memory usage
    mem = process.memory_info().rss / (1024 * 1024)
    print(f"\r[Memory] RSS: {mem:.2f} MB", flush=True)

def chunk_text_file(file_path: str, num_processes: int, special_token: str) -> Counter:
    f: BufferedReader = open(file_path, 'rb')
    content = f.read().decode("utf-8")
    f.close()

    chunk_starts = [0]
    start_index = 0
    while True:
        match_index = content.find(special_token, start_index)
        if match_index == -1:
            break
        chunk_starts.append(match_index)
        start_index = match_index + len(special_token)
    chunk_ends = chunk_starts[1:]
    chunk_ends.append(len(content))
    chunk_boundaries = list(zip(chunk_starts, chunk_ends))

    partial_pretokenize = functools.partial(pretokenize_chunk, content=content, special_token=special_token)

    # Start memory reporter thread
    stop_event = threading.Event()
    mem_thread = threading.Thread(target=memory_reporter, args=(stop_event,), daemon=True)
    mem_thread.start()

    # Use tqdm for progress bar
    results = []
    with multiprocessing.Pool(num_processes) as p:
        for counter in tqdm(p.imap(partial_pretokenize, chunk_boundaries), total=len(chunk_boundaries), desc="Pretokenizing"):
            results.append(counter)
        p.close()
        p.join()

    stop_event.set()
    mem_thread.join()

    final_pretoken_frequency_counts = sum(results, Counter())
    print()  # For clean output after progress/memory

    final_pretoken_frequency_counts
    return final_pretoken_frequency_counts

def train_bpe(pretokenized_freq: Counter[str], num_merges: int):
    # Represent every token as a list[int] so we can edit in place
    corpus = {tuple(token.encode()): count for token, count in pretokenized_freq.items()}
    vocab = {tuple([i]): i for i in range(256)}      # byte → id
    next_id = 256
    merges: dict[tuple[int, int], int] = {}

    for _ in trange(num_merges, desc="BPE merges"):
        # 1. Count adjacent pairs
        pair_counts = defaultdict(int)
        for symbols, freq in corpus.items():
            for a, b in zip(symbols[:-1], symbols[1:]):
                pair_counts[(a, b)] += freq

        if not pair_counts:                 # nothing left to merge
            break

        most_common = max(pair_counts, key=pair_counts.get)

        # 2. Add new symbol to vocab & merges table
        merges[most_common] = next_id
        vocab[most_common] = next_id
        new_symbol = next_id
        next_id += 1

        # 3. Replace occurrences in every token
        updated_corpus = {}
        for symbols, freq in corpus.items():
            merged = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == most_common:
                    merged.append(new_symbol)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            updated_corpus[tuple(merged)] = freq
        corpus = updated_corpus

    return merges, vocab



if __name__ == "__main__":
    pretokenized_frequency_table = chunk_text_file("data/toy_data.txt", 8, "<|endoftext|>")
    print(train_bpe(pretokenized_frequency_table, 1000))

