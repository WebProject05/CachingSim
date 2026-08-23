# Core cache model

This package owns file metadata, cache memory accounting, eviction by utility, and the state vector consumed by decision-making code.

`CacheEngine` tracks cached status, used capacity, current simulation time, and request popularity in a sliding window. `GetCurrentState` assembles memory, popularity, utility, file-size, and cache-status vectors for the configured number of file types.
