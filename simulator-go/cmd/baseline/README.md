# Baseline comparison

`main.go` creates deterministic files and one Zipf request trace, then replays that identical trace through FIFO, LRU, LFU, and SIEVE. This keeps algorithm comparisons fair.

Useful flags include `-requests`, `-files`, `-capacity`, `-eta`, `-seed`, `-window`, `-log`, and `-graphs`. Every invocation appends a compact record to `logs/baseline_runs.log`: shared configuration plus six comparison metrics per algorithm. The terminal still shows the complete report. Parent directories are created automatically; use `-log path/to/other.log` to select another file.

Each run replaces the `graphs` directory (or the path passed to `-graphs`) and writes:

- `results.csv`: one row per algorithm with hit count, hit rates, utility, and run parameters.
- `cumulative_hit_rate.svg`: trial versus cumulative hit rate, one line per algorithm.
- `hit_count_vs_cache_size.svg`: cache size versus hit count.
- `hit_count_vs_request_rate.svg`: lambda request rate versus hit count.
- `hit_count_vs_zipf_eta.svg`: Zipf eta versus hit count.
- `hit_count_vs_file_lifetime.svg`: most popular file lifetime versus hit count.
- `hit_count_vs_file_size.svg`: most popular file size versus hit count.
- `total_utility_vs_cache_size.svg`: cache size versus total utility.
- `sweep_parameters.csv`: the parameter values used for each sweep.

All SVG charts are line graphs with one line for FIFO, LRU, LFU, and SIEVE. Use `-graph-interval` to change convergence sampling, and use comma-separated flags such as `-cache-sizes`, `-request-rates`, `-zipf-etas`, `-file-lifetimes`, and `-file-sizes` to change sweep points.

The SVG files open directly in a browser or can be imported into a report. To keep multiple experiments, copy the generated directory after each run or provide a separate `-graphs` path.

The first report is a compact comparison table. Detailed sections include hit and miss rates, byte hit rate, rejection and eviction rates, insertions, occupancy, traffic volume, runtime throughput, and total utility.

Example:

```text
go run ./cmd/baseline -requests 50000 -seed 42 -graphs graphs
```
