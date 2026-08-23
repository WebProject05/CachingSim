# Go Simulator

This directory contains the Go implementation of the edge-caching simulator.

## Execution paths

- `go run ./cmd/baseline` generates one request trace and compares LRU, LFU, and SIEVE on that same trace.
- `go run ./cmd/server` starts the gRPC environment server used by external agents.
- `go test ./...` runs cache, reward, and baseline tests.

The simulator uses MiB for file sizes and cache capacity. Use `-seed` on the baseline command to make a run reproducible. The comparison output reports request-level hit rate, byte hit rate, cache churn, occupancy, and timing.

## Package flow

`config` defines experiment parameters. `core` owns file metadata and cache state. `smdp` produces arrivals and requests. `baselines` implements replacement policies and their measurements. `server` exposes the environment over gRPC, while `pb` contains generated protobuf bindings. `cmd` contains executable entry points and `tests` contains cross-package tests.
