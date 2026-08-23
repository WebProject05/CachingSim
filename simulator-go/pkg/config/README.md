# Configuration

`config.go` defines the shared `Config` structure and `DefaultConfig`. It controls file count, cache capacity, sliding-window size, utility bounds, discounting, arrival rates, and Zipf popularity.

Keep units explicit: capacities and file sizes are MiB, while arrival rates and utility parameters follow the simulator model. Commands may copy and adjust a default configuration for an experiment.
