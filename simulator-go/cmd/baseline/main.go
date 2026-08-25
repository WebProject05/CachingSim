package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"smdp-edge-caching-framework/pkg/baselines"
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
	"smdp-edge-caching-framework/pkg/smdp"
)

var output io.Writer = os.Stdout

type cache interface {
	Access(int) bool
	Stats() baselines.CacheStats
}
type runResult struct {
	name                                                         string
	stats                                                        baselines.CacheStats
	elapsedSeconds, warmupMissRate, steadyMissRate, totalUtility float64
	cumulativeTrials, cumulativeMisses                           []float64
}
type requestEvent struct {
	fileID int
	time   float64
}

func main() {
	requests := flag.Int("requests", 50000, "number of requests in the experiment")
	fileCount := flag.Int("files", 50, "number of generated files")
	capacity := flag.Float64("capacity", 10000, "cache capacity in MiB")
	eta := flag.Float64("eta", 1.0, "Zipf popularity exponent")
	seed := flag.Int64("seed", 42, "random seed used for files and requests")
	window := flag.Int("window", 0, "warm-up and final-window size; 0 means 10% of the trace")
	logPath := flag.String("log", "logs/baseline_runs.log", "append-only log file")
	graphsPath := flag.String("graphs", "graphs", "directory replaced with graph output")
	graphInterval := flag.Int("graph-interval", 100, "trials between convergence graph points")
	concurrent := flag.Bool("concurrent", false, "run cache algorithms concurrently; default is sequential")
	cacheSizes := flag.String("cache-sizes", "1000,5000,10000,20000,50000", "cache sizes in MiB")
	requestRates := flag.String("request-rates", "0.05,0.1,0.2,0.5,1.0", "lambda request rates")
	zipfEtas := flag.String("zipf-etas", "0.2,0.5,1.0,1.5,2.0", "Zipf eta values")
	fileLifetimes := flag.String("file-lifetimes", "10,15,20,25,30", "popular-file lifetime values")
	fileSizes := flag.String("file-sizes", "100,300,500,700,900", "popular-file sizes in MiB")
	flag.Parse()
	if err := prepareGraphsDirectory(*graphsPath); err != nil {
		fmt.Fprintf(os.Stderr, "could not prepare graph directory: %v\n", err)
		return
	}
	logFile, err := openRunLog(*logPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "could not open run log: %v\n", err)
		return
	}
	defer logFile.Close()
	if *requests < 1 || *fileCount < 1 || *capacity <= 0 {
		message := "requests and files must be positive; capacity must be greater than zero"
		fmt.Fprintln(output, message)
		writeInvalidRunLog(logFile, *logPath, *seed, *requests, *fileCount, *capacity, *eta, message)
		return
	}
	if *graphInterval < 1 {
		*graphInterval = 1
	}
	cfg := config.DefaultConfig()
	cfg.TotalFileTypes, cfg.CacheCapacity, cfg.ZipfEta = *fileCount, *capacity, *eta
	files := core.GenerateFiles(cfg, rand.New(rand.NewSource(*seed)))
	events := generateRequestEvents(cfg, *seed, *requests)
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
	printBox("FIFO / LRU / LFU / SIEVE CACHE EXPERIMENT", []string{fmt.Sprintf("Seed              : %d", *seed), fmt.Sprintf("Requests          : %d", *requests), fmt.Sprintf("Files             : %d", *fileCount), fmt.Sprintf("Capacity          : %.2f MiB", *capacity), fmt.Sprintf("Zipf eta          : %.3f", cfg.ZipfEta), fmt.Sprintf("Lambda source     : %.3f", cfg.LambdaSource), fmt.Sprintf("Measurement window: %d", measurementWindow)})
	printFileStats(files)
	results := runAllWithConcurrency(cfg, files, events, measurementWindow, *graphInterval, *concurrent)
	printComparisonTable(results)
	for _, result := range results {
		printResult(result)
	}
	if err := writeGraphs(*graphsPath, *seed, *requests, *fileCount, *capacity, cfg.ZipfEta, *graphInterval, *cacheSizes, *requestRates, *zipfEtas, *fileLifetimes, *fileSizes, cfg, files, events, results, *concurrent); err != nil {
		fmt.Fprintf(os.Stderr, "could not write graphs: %v\n", err)
		return
	}
	writeRunLog(logFile, *logPath, *seed, *requests, *fileCount, *capacity, cfg.ZipfEta, measurementWindow, *concurrent, results)
}

func generateRequestEvents(cfg *config.Config, seed int64, requests int) []requestEvent {
	generator := smdp.NewGenerator(cfg, seed)
	events := make([]requestEvent, requests)
	currentTime := 0.0
	for index := range events {
		currentTime += generator.NextPoissonInterval(cfg.LambdaSource)
		events[index] = requestEvent{generator.SampleRequestedFile(), currentTime}
	}
	return events
}

func runAll(cfg *config.Config, files []core.FileMetadata, events []requestEvent, window, interval int) []runResult {
	return runAllWithConcurrency(cfg, files, events, window, interval, false)
}

func runAllWithConcurrency(cfg *config.Config, files []core.FileMetadata, events []requestEvent, window, interval int, concurrent bool) []runResult {
	algorithms := []struct {
		name     string
		newCache func() cache
	}{
		{"FIFO", func() cache { return baselines.NewFIFOCache(cfg, files) }},
		{"LRU", func() cache { return baselines.NewLRUCache(cfg, files) }},
		{"LFU", func() cache { return baselines.NewLFUCache(cfg, files) }},
		{"SIEVE", func() cache { return baselines.NewSIEVECache(cfg, files) }},
	}
	results := make([]runResult, len(algorithms))
	if !concurrent {
		for index, algorithm := range algorithms {
			results[index] = run(algorithm.name, algorithm.newCache, events, files, cfg, window, interval)
		}
		return results
	}

	var waitGroup sync.WaitGroup
	waitGroup.Add(len(algorithms))
	for index, algorithm := range algorithms {
		go func(index int, algorithm struct {
			name     string
			newCache func() cache
		}) {
			defer waitGroup.Done()
			results[index] = run(algorithm.name, algorithm.newCache, events, files, cfg, window, interval)
		}(index, algorithm)
	}
	waitGroup.Wait()
	return results
}

func run(name string, newCache func() cache, events []requestEvent, files []core.FileMetadata, cfg *config.Config, window, interval int) runResult {
	c := newCache()
	warmup, steady, hits := 0, 0, 0
	utility := 0.0
	cumulative := make([]float64, 0, (len(events)+interval-1)/interval)
	cumulativeTrials := make([]float64, 0, cap(cumulative))
	start := time.Now()
	for index, event := range events {
		if c.Access(event.fileID) {
			hits++
			if event.fileID >= 0 && event.fileID < len(files) {
				utility += files[event.fileID].Utility(event.time, cfg)
			}
		} else {
			if index < window {
				warmup++
			}
			if index >= len(events)-window {
				steady++
			}
		}
		if (index+1)%interval == 0 || index == len(events)-1 {
			cumulativeTrials = append(cumulativeTrials, float64(index+1))
			cumulative = append(cumulative, 100-float64(hits)/float64(index+1)*100)
		}
	}
	return runResult{name, c.Stats(), time.Since(start).Seconds(), float64(warmup) / float64(window), float64(steady) / float64(window), utility, cumulativeTrials, cumulative}
}

func formatTiming(result runResult) (string, string) {
	if result.stats.Requests == 0 || result.elapsedSeconds <= 0 {
		return "n/a", "n/a"
	}
	return fmt.Sprintf("%.0f", float64(result.stats.Requests)/result.elapsedSeconds), fmt.Sprintf("%.1f", result.elapsedSeconds*float64(time.Second)/float64(result.stats.Requests))
}

func printResult(result runResult) {
	s := result.stats
	ops, latency := formatTiming(result)
	printBox(result.name, []string{fmt.Sprintf("Requests          : %d", s.Requests), fmt.Sprintf("Hits              : %d", s.Hits), fmt.Sprintf("Misses            : %d", s.Misses), fmt.Sprintf("Hit rate          : %.2f%%", s.HitRate()*100), fmt.Sprintf("Byte hit rate     : %.2f%%", s.ByteHitRate()*100), fmt.Sprintf("Evictions         : %d", s.Evictions), fmt.Sprintf("Utilization       : %.2f%%", s.Utilization()*100), fmt.Sprintf("Total utility     : %.4f", result.totalUtility), fmt.Sprintf("Average ns/request: %s", latency), fmt.Sprintf("Operations/sec    : %s", ops), fmt.Sprintf("Warm-up miss rate : %.2f%%", result.warmupMissRate*100), fmt.Sprintf("Final miss rate   : %.2f%%", result.steadyMissRate*100)})
}

func printComparisonTable(results []runResult) {
	printBox("ALGORITHM COMPARISON", []string{"Higher hit rate and byte hit rate are better.", "Lower miss, rejection, eviction, and latency values are better."})
	border := "+------------+--------+--------+--------+--------+--------+--------+"
	fmt.Fprintln(output, border)
	fmt.Fprintln(output, "| Algorithm  | Hit %   | Byte %  | Miss %  | Evict % | Util %  | Ops/sec |")
	fmt.Fprintln(output, border)
	for _, r := range results {
		ops, _ := formatTiming(r)
		s := r.stats
		fmt.Fprintf(output, "| %-10s | %6.2f%% | %6.2f%% | %6.2f%% | %6.2f%% | %6.2f%% | %7s |\n", r.name, s.HitRate()*100, s.ByteHitRate()*100, s.MissRate()*100, s.EvictionRate()*100, s.Utilization()*100, ops)
	}
	fmt.Fprintln(output, border)
}

func openRunLog(path string) (*os.File, error) {
	if directory := filepath.Dir(path); directory != "." {
		if err := os.MkdirAll(directory, 0755); err != nil {
			return nil, err
		}
	}
	return os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
}

func writeRunLog(w io.Writer, path string, seed int64, requests, files int, capacity, eta float64, window int, concurrent bool, results []runResult) {
	fmt.Fprintf(w, "RUN START : %s\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(w, "LOG FILE  : %s\n", path)
	fmt.Fprintf(w, "CONFIG    : seed=%d requests=%d files=%d capacity=%.2f MiB eta=%.3f window=%d concurrent=%t\n", seed, requests, files, capacity, eta, window, concurrent)
	fmt.Fprintln(w, "METRICS   : hit_rate | byte_hit_rate | miss_rate | eviction_rate | utilization | avg_ns/request")
	for _, r := range results {
		_, latency := formatTiming(r)
		fmt.Fprintf(w, "%-8s : %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %s\n",
			r.name,
			r.stats.HitRate()*100,
			r.stats.ByteHitRate()*100,
			r.stats.MissRate()*100,
			r.stats.EvictionRate()*100,
			r.stats.Utilization()*100,
			latency)
	}
	fmt.Fprintf(w, "RUN END   : %s\n\n", time.Now().Format(time.RFC3339))
}

func writeInvalidRunLog(w io.Writer, path string, seed int64, requests, files int, capacity, eta float64, message string) {
	fmt.Fprintf(w, "RUN START : %s\nLOG FILE  : %s\nCONFIG    : seed=%d requests=%d files=%d capacity=%.2f MiB eta=%.3f\nSTATUS    : invalid - %s\nRUN END   : %s\n\n", time.Now().Format(time.RFC3339), path, seed, requests, files, capacity, eta, message, time.Now().Format(time.RFC3339))
}

func printFileStats(files []core.FileMetadata) {
	minimum, maximum, total := math.Inf(1), math.Inf(-1), 0.0
	for _, file := range files {
		minimum = math.Min(minimum, file.Size)
		maximum = math.Max(maximum, file.Size)
		total += file.Size
	}
	printBox("GENERATED FILES", []string{fmt.Sprintf("Count             : %d", len(files)), fmt.Sprintf("Total size        : %.2f MiB", total), fmt.Sprintf("Average size      : %.2f MiB", total/float64(len(files))), fmt.Sprintf("Minimum size      : %.2f MiB", minimum), fmt.Sprintf("Maximum size      : %.2f MiB", maximum)})
}

func printBox(title string, lines []string) {
	width := len(title) + 4
	for _, line := range lines {
		if len(line)+4 > width {
			width = len(line) + 4
		}
	}
	border := "+" + strings.Repeat("-", width-2) + "+"
	fmt.Fprintln(output, border)
	fmt.Fprintf(output, "| %-*s |\n", width-4, title)
	fmt.Fprintln(output, border)
	for _, line := range lines {
		fmt.Fprintf(output, "| %-*s |\n", width-4, line)
	}
	fmt.Fprintln(output, border)
}
