import regex

# Keep the Rust default_pattern in sync with this GPT-2 pretokenizer.
PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKENIZER = regex.compile(PRETOKENIZER_PATTERN)
