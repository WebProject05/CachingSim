package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"smdp-edge-caching-framework/pkg/baselines"
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
	"smdp-edge-caching-framework/pkg/smdp"
)

type cache interface {
	Access(fileID int) bool
	Stats() baselines.CacheStats
}

type runResult struct {
	name           string
	stats          baselines.CacheStats
	elapsedSeconds float64
	warmupMissRate float64
	steadyMissRate float64
}

func main() {
	requests := flag.Int("requests", 50000, "number of requests in the experiment")
	fileCount := flag.Int("files", 50, "number of generated files")
	capacity := flag.Float64("capacity", 10000, "cache capacity in MiB")
	eta := flag.Float64("eta", 1.0, "Zipf popularity exponent")
	seed := flag.Int64("seed", 42, "random seed used for files and requests")
	window := flag.Int("window", 0, "requests used for warm-up and final-window miss rates; 0 means 10% of the trace")
	flag.Parse()

	if *requests < 1 || *fileCount < 1 || *capacity <= 0 {
		fmt.Println("requests and files must be positive; capacity must be greater than zero")
		return
	}

	cfg := config.DefaultConfig()
	cfg.TotalFileTypes = *fileCount
	cfg.CacheCapacity = *capacity
	cfg.ZipfEta = *eta

	files := core.GenerateFiles(cfg, rand.New(rand.NewSource(*seed)))
	generator := smdp.NewGenerator(cfg, *seed)
	requestTrace := make([]int, *requests)
	for index := range requestTrace {
		requestTrace[index] = generator.SampleRequestedFile()
	}

	measurementWindow := *window
	if measurementWindow <= 0 {
		measurementWindow = *requests / 10
		if measurementWindow == 0 {
			measurementWindow = 1
		}
	}
	if measurementWindow > *requests {
		measurementWindow = *requests
	}

	printBox("LRU / LFU / SIEVE CACHE EXPERIMENT", []string{
		fmt.Sprintf("Seed              : %d", *seed),
		fmt.Sprintf("Requests          : %d", *requests),
		fmt.Sprintf("Files             : %d", *fileCount),
		fmt.Sprintf("Capacity          : %.2f MiB", *capacity),
		fmt.Sprintf("Zipf eta          : %.3f", cfg.ZipfEta),
		fmt.Sprintf("Lambda source     : %.3f", cfg.LambdaSource),
		fmt.Sprintf("Measurement window: %d", measurementWindow),
	})
	printFileStats(files)

	results := []runResult{
		run("LRU", func() cache { return baselines.NewLRUCache(cfg, files) }, requestTrace, measurementWindow),
		run("LFU", func() cache { return baselines.NewLFUCache(cfg, files) }, requestTrace, measurementWindow),
		run("SIEVE", func() cache { return baselines.NewSIEVECache(cfg, files) }, requestTrace, measurementWindow),
	}

	printComparisonTable(results)
	for _, result := range results {
		stats := result.stats
		operationsPerSecond, averageNanoseconds := formatTiming(result)
		printBox(result.name, []string{
			fmt.Sprintf("Requests          : %d", stats.Requests),
			fmt.Sprintf("Hits              : %d", stats.Hits),
			fmt.Sprintf("Misses            : %d", stats.Misses),
			fmt.Sprintf("Hit rate          : %.2f%%", stats.HitRate()*100),
			fmt.Sprintf("Miss rate         : %.2f%%", stats.MissRate()*100),
			fmt.Sprintf("Byte hit rate     : %.2f%%", stats.ByteHitRate()*100),
			fmt.Sprintf("Evictions         : %d", stats.Evictions),
			fmt.Sprintf("Eviction rate     : %.2f%%", stats.EvictionRate()*100),
			fmt.Sprintf("Rejected          : %d (%.2f%%)", stats.RejectedRequests, stats.RejectionRate()*100),
			fmt.Sprintf("Insertions        : %d", stats.Insertions),
			fmt.Sprintf("Cached files      : %d", stats.CachedFiles),
			fmt.Sprintf("Used capacity     : %.2f / %.2f MiB", stats.UsedCapacity, stats.Capacity),
			fmt.Sprintf("Utilization       : %.2f%%", stats.Utilization()*100),
			fmt.Sprintf("Free capacity     : %.2f MiB", stats.Capacity-stats.UsedCapacity),
			fmt.Sprintf("Requested bytes   : %.2f MiB", stats.RequestedBytes),
			fmt.Sprintf("Hit bytes         : %.2f MiB", stats.HitBytes),
			fmt.Sprintf("Miss bytes        : %.2f MiB", stats.MissBytes),
			fmt.Sprintf("Average request   : %.2f MiB", stats.AverageRequestBytes()),
			fmt.Sprintf("Average ns/request: %s", averageNanoseconds),
			fmt.Sprintf("Operations/sec    : %s", operationsPerSecond),
			fmt.Sprintf("Warm-up miss rate : %.2f%%", result.warmupMissRate*100),
			fmt.Sprintf("Final miss rate   : %.2f%%", result.steadyMissRate*100),
			fmt.Sprintf("Miss-rate change  : %.2f points", (result.steadyMissRate-result.warmupMissRate)*100),
		})
	}
}

func printComparisonTable(results []runResult) {
	printBox("ALGORITHM COMPARISON", []string{
		"Higher hit rate and byte hit rate are better.",
		"Lower miss, rejection, eviction, and latency values are better.",
	})
	fmt.Println("Algorithm     Hit rate  Byte hit  Miss rate  Reject  Evict  Ops/sec  Avg ns/req")
	fmt.Println("------------  --------  --------  ---------  ------  -----  -------  -----------")
	for _, result := range results {
		stats := result.stats
		operationsPerSecond, averageNanoseconds := formatTiming(result)
		fmt.Printf("%-12s  %7.2f%%  %7.2f%%  %8.2f%%  %5.2f%%  %5.2f%%  %7s  %11s\n",
			result.name,
			stats.HitRate()*100,
			stats.ByteHitRate()*100,
			stats.MissRate()*100,
			stats.RejectionRate()*100,
			stats.EvictionRate()*100,
			operationsPerSecond,
			averageNanoseconds,
		)
	}
	fmt.Println()
}

func formatTiming(result runResult) (operationsPerSecond string, averageNanoseconds string) {
	if result.stats.Requests == 0 || result.elapsedSeconds <= 0 {
		return "n/a", "n/a"
	}
	return fmt.Sprintf("%.0f", float64(result.stats.Requests)/result.elapsedSeconds),
		fmt.Sprintf("%.1f", result.elapsedSeconds*float64(time.Second)/float64(result.stats.Requests))
}

func run(name string, newCache func() cache, requestTrace []int, measurementWindow int) runResult {
	cache := newCache()
	warmupMisses := 0
	steadyMisses := 0
	start := time.Now()
	for index, fileID := range requestTrace {
		if !cache.Access(fileID) {
			if index < measurementWindow {
				warmupMisses++
			}
			if index >= len(requestTrace)-measurementWindow {
				steadyMisses++
			}
		}
	}
	return runResult{
		name:           name,
		stats:          cache.Stats(),
		elapsedSeconds: time.Since(start).Seconds(),
		warmupMissRate: float64(warmupMisses) / float64(measurementWindow),
		steadyMissRate: float64(steadyMisses) / float64(measurementWindow),
	}
}

func printFileStats(files []core.FileMetadata) {
	minimum := math.Inf(1)
	maximum := math.Inf(-1)
	total := 0.0
	for _, file := range files {
		minimum = math.Min(minimum, file.Size)
		maximum = math.Max(maximum, file.Size)
		total += file.Size
	}
	printBox("GENERATED FILES", []string{
		fmt.Sprintf("Count             : %d", len(files)),
		fmt.Sprintf("Total size        : %.2f MiB", total),
		fmt.Sprintf("Average size      : %.2f MiB", total/float64(len(files))),
		fmt.Sprintf("Minimum size      : %.2f MiB", minimum),
		fmt.Sprintf("Maximum size      : %.2f MiB", maximum),
	})
}

func printBox(title string, lines []string) {
	width := len(title) + 4
	for _, line := range lines {
		if len(line)+4 > width {
			width = len(line) + 4
		}
	}
	border := "+" + strings.Repeat("-", width-2) + "+"
	fmt.Println(border)
	fmt.Printf("| %-*s |\n", width-4, title)
	fmt.Println(border)
	for _, line := range lines {
		fmt.Printf("| %-*s |\n", width-4, line)
	}
	fmt.Println(border)
}
