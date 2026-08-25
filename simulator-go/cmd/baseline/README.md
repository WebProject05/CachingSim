# Baseline comparison

`main.go` creates deterministic files and one Zipf request trace, then replays that identical trace through FIFO, LRU, LFU, and SIEVE. This keeps algorithm comparisons fair.

Useful flags include `-requests`, `-files`, `-capacity`, `-eta`, `-seed`, `-window`, `-log`, and `-graphs`. Every invocation appends a compact record to `logs/baseline_runs.log`: shared configuration plus the six metrics listed in the `METRICS` line for each algorithm. The terminal still shows the complete report. Parent directories are created automatically; use `-log path/to/other.log` to select another file.

Each run replaces the `graphs` directory (or the path passed to `-graphs`) and writes:

- `results.csv`: one row per algorithm with hit count, hit rates, utility, and run parameters.
- `cumulative_miss_rate.svg`: trial versus cumulative miss rate, one line per algorithm. Lower and flatter lines indicate better performance and stabilization.
- `miss_rate_vs_cache_size.svg`: cache size versus miss ratio percentage.
- `miss_rate_vs_request_rate.svg`: lambda request rate versus miss ratio percentage.
- `miss_rate_vs_zipf_eta.svg`: Zipf eta versus miss ratio percentage.
- `miss_rate_vs_file_lifetime.svg`: most popular file lifetime versus miss ratio percentage.
- `miss_rate_vs_file_size.svg`: most popular file size versus miss ratio percentage.
- `total_utility_vs_cache_size.svg`: cache size versus total utility.
- `sweep_parameters.csv`: the parameter values used for each sweep.

All SVG charts are line graphs with one line for FIFO, LRU, LFU, and SIEVE. Use `-graph-interval` to change convergence sampling, and use comma-separated flags such as `-cache-sizes`, `-request-rates`, `-zipf-etas`, `-file-lifetimes`, and `-file-sizes` to change sweep points.

The baseline command is sequential by default. Pass `-concurrent=true` to opt in to running FIFO, LRU, LFU, and SIEVE in separate goroutines. The concurrent mode uses the same generated files and request trace for every algorithm, preserves the documented algorithm order in results, and waits for all algorithm runs before writing reports. It also applies to each individual point in graph sweeps; sweep points and graph file writes remain sequential. This mode can reduce elapsed time when CPU cores are available, but can increase CPU and memory usage, and timing results may vary between runs. Use `-concurrent=false` explicitly when a sequential run is required. The selected mode is recorded in the run log.

The SVG files open directly in a browser or can be imported into a report. To keep multiple experiments, copy the generated directory after each run or provide a separate `-graphs` path.

The first report is a compact comparison table. Detailed sections include hit and miss rates, byte hit rate, rejection and eviction rates, insertions, occupancy, traffic volume, runtime throughput, and total utility.

Example:

```text
go run ./cmd/baseline -requests 50000 -seed 42 -graphs graphs
```

Opt-in concurrent example:

```text
go run ./cmd/baseline -requests 50000 -seed 42 -concurrent=true -graphs graphs-concurrent
```

Keep each flag as a separate argument. In particular, write `-seed 42 -concurrent` with a space; `-seed 42-concurrent` is invalid because the seed must be an integer. Since `concurrent` is a boolean flag, both `-concurrent` and `-concurrent=true` enable it.
