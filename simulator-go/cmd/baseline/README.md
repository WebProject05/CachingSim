# Baseline comparison

`main.go` creates deterministic files and one Zipf request trace, then replays that identical trace through LRU, LFU, and SIEVE. This keeps algorithm comparisons fair.

Useful flags include `-requests`, `-files`, `-capacity`, `-eta`, `-seed`, and `-window`. The first report is a compact comparison table. Detailed sections include hit and miss rates, byte hit rate, rejection and eviction rates, insertions, occupancy, traffic volume, and runtime throughput.

Example:

```text
go run ./cmd/baseline -requests 50000 -seed 42
```
