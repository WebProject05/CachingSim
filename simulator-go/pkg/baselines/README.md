# Baseline policies

This package provides replacement-policy implementations used for repeatable comparisons:

- LRU evicts the least recently used cached file.
- LFU evicts the least frequently used file, breaking ties by oldest use.
- SIEVE uses a reference bit and a moving hand to find eviction candidates.

Each policy exposes `Access(fileID) bool` and `Stats() CacheStats`. A hit returns `true`; a miss, invalid request, or rejected file returns `false`. `CacheStats` includes request counts, capacity, byte traffic, insertions, and derived rates.
