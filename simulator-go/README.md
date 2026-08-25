# Go Simulator

This directory contains the Go implementation of the edge-caching simulator.

## Execution paths

- `go run ./cmd/baseline` generates one request trace and compares FIFO, LRU, LFU, and SIEVE on that same trace.
- `go run ./cmd/server` starts the gRPC environment server used by external agents.
- `go test ./...` runs cache, reward, and baseline tests.

The simulator uses MiB for file sizes and cache capacity. Use `-seed` on the baseline command to make a run reproducible. The comparison output reports request-level hit rate, byte hit rate, cache churn, occupancy, and timing. A compact baseline summary is appended to `logs/baseline_runs.log` on every invocation.

The baseline command is sequential by default. Pass `-concurrent=true` to opt in to running the four cache algorithms in separate goroutines using the same trace. This preserves result order and applies to algorithm runs inside graph sweeps; it may use more CPU and memory. The log records the selected mode. Use `-concurrent=false` explicitly when a sequential run is required.

Flags must be separated by spaces. Use `-seed 42 -concurrent` or `-seed 42 -concurrent=true`; do not combine them as `-seed 42-concurrent`.

## Package flow

`config` defines experiment parameters. `core` owns file metadata and cache state. `smdp` produces arrivals and requests. `baselines` implements replacement policies and their measurements. `server` exposes the environment over gRPC, while `pb` contains generated protobuf bindings. `cmd` contains executable entry points and `tests` contains cross-package tests.
