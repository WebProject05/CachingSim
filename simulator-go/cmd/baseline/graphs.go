package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"smdp-edge-caching-framework/pkg/baselines"
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

type lineSeries struct {
	name string
	x, y []float64
}

type sweepSpec struct {
	file, title, xLabel, yLabel string
	values                      []float64
	isUtilityMetric             bool
	metric                      string
	configure                   func(*config.Config, []core.FileMetadata, float64) ([]core.FileMetadata, *config.Config)
}

func prepareGraphsDirectory(path string) error {
	if err := os.RemoveAll(path); err != nil {
		return err
	}
	return os.MkdirAll(path, 0755)
}

func writeGraphs(
	path string,
	seed int64,
	requests, fileCount int,
	capacity, eta float64,
	interval int,
	cacheSizeText, rateText, etaText, lifetimeText, sizeText, importanceText string,
	cfg *config.Config,
	files []core.FileMetadata,
	events []requestEvent,
	results []runResult,
	concurrent bool,
) error {
	// 1. Write CSV and JSON baseline run results
	if err := writeResultsCSV(path, seed, requests, fileCount, capacity, eta, results); err != nil {
		return err
	}
	if err := writeResultsJSON(path, seed, requests, fileCount, capacity, eta, cfg.LambdaSource, results); err != nil {
		return err
	}

	// 2. Write Convergence Line Chart (AdaLife-Pro vs Paper-Proposed ONLY; CTD omitted from graphs)
	var paperResults []runResult
	for _, r := range results {
		if r.name == "AdaLife-Pro" || r.name == "SMDP-DDQL" || r.name == "Paper-Proposed" {
			displayName := r.name
			if r.name == "SMDP-DDQL" || r.name == "AdaLife-Pro" {
				displayName = "AdaLife-Pro (Ours)"
			} else if r.name == "Paper-Proposed" {
				displayName = "Paper (Niknia '24 DDQL)"
			}
			copyResult := r
			copyResult.name = displayName
			paperResults = append(paperResults, copyResult)
		}
	}
	if err := writeLineChart(filepath.Join(path, "cumulative_miss_rate.svg"), "Miss-rate convergence", "Trial", "Cumulative miss rate (%)", convergenceSeries(paperResults, interval)); err != nil {
		return err
	}

	// 3. Write Elastic Capacity Trajectory SVG and CSV
	if err := writeElasticCapacityArtifacts(path, results, interval, cfg.CacheCapacity); err != nil {
		return err
	}

	specs := []sweepSpec{
		// Fig. 6c & 7c: Cache size sweeps
		{"hit_count_vs_cache_size.svg", "Total hit count vs cache size (Fig. 6c)", "Cache size (MiB)", "Total hit count", parseValues(cacheSizeText), false, "hits",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.CacheCapacity = v
				return f, &n
			}},
		{"total_utility_vs_cache_size.svg", "Total utility vs cache size (Fig. 7c)", "Cache size (MiB)", "Total utility", parseValues(cacheSizeText), true, "utility",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.CacheCapacity = v
				return f, &n
			}},
		{"evictions_vs_cache_size.svg", "Total evictions vs cache size (Churn Analysis)", "Cache size (MiB)", "Total evictions", parseValues(cacheSizeText), false, "evictions",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.CacheCapacity = v
				return f, &n
			}},
		{"byte_hit_rate_vs_cache_size.svg", "Byte hit rate vs cache size (Bandwidth Savings)", "Cache size (MiB)", "Byte hit rate (%)", parseValues(cacheSizeText), false, "byte_hit_rate",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.CacheCapacity = v
				return f, &n
			}},

		// Fig. 6b & 7b: Request rate lambda sweeps
		{"hit_count_vs_request_rate.svg", "Total hit count vs request rate (Fig. 6b)", "Request rate lambda", "Total hit count", parseValues(rateText), false, "hits",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.LambdaSource = v
				return f, &n
			}},
		{"total_utility_vs_request_rate.svg", "Total utility vs request rate (Fig. 7b)", "Request rate lambda", "Total utility", parseValues(rateText), true, "utility",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.LambdaSource = v
				return f, &n
			}},
		{"evictions_vs_request_rate.svg", "Total evictions vs request rate (Load Churn)", "Request rate lambda", "Total evictions", parseValues(rateText), false, "evictions",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.LambdaSource = v
				return f, &n
			}},
		{"byte_hit_rate_vs_request_rate.svg", "Byte hit rate vs request rate", "Request rate lambda", "Byte hit rate (%)", parseValues(rateText), false, "byte_hit_rate",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.LambdaSource = v
				return f, &n
			}},

		// Fig. 6a & 7a: Popularity skewness eta sweeps
		{"hit_count_vs_zipf_eta.svg", "Total hit count vs popularity skewness (Fig. 6a)", "Zipf eta", "Total hit count", parseValues(etaText), false, "hits",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.ZipfEta = v
				return f, &n
			}},
		{"total_utility_vs_zipf_eta.svg", "Total utility vs popularity skewness (Fig. 7a)", "Zipf eta", "Total utility", parseValues(etaText), true, "utility",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				n := *c
				n.ZipfEta = v
				return f, &n
			}},

		// Fig. 4a & 5a: Lifetime of popular file sweeps (all other files fixed: lifetime=20, size=500, importance=0.5)
		{"hit_count_vs_file_lifetime.svg", "Total hit count vs popular-file lifetime (Fig. 4a)", "Popular file lifetime", "Total hit count", parseValues(lifetimeText), false, "hits",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				return core.GenerateSweepFiles(c, v, 500.0, 0.5), c
			}},
		{"total_utility_vs_file_lifetime.svg", "Total utility vs popular-file lifetime (Fig. 5a)", "Popular file lifetime", "Total utility", parseValues(lifetimeText), true, "utility",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				return core.GenerateSweepFiles(c, v, 500.0, 0.5), c
			}},

		// Fig. 4b & 5b: Size of popular file sweeps (all other files fixed: size=500, lifetime=20, importance=0.5)
		{"hit_count_vs_file_size.svg", "Total hit count vs popular-file size (Fig. 4b)", "Popular file size (MiB)", "Total hit count", parseValues(sizeText), false, "hits",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				return core.GenerateSweepFiles(c, 20.0, v, 0.5), c
			}},
		{"total_utility_vs_file_size.svg", "Total utility vs popular-file size (Fig. 5b)", "Popular file size (MiB)", "Total utility", parseValues(sizeText), true, "utility",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				return core.GenerateSweepFiles(c, 20.0, v, 0.5), c
			}},

		// Fig. 4c & 5c: Importance of popular file sweeps (all other files fixed: importance=0.5, lifetime=20, size=500)
		{"hit_count_vs_file_importance.svg", "Total hit count vs popular-file importance (Fig. 4c)", "Popular file importance", "Total hit count", parseValues(importanceText), false, "hits",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				return core.GenerateSweepFiles(c, 20.0, 500.0, v), c
			}},
		{"total_utility_vs_file_importance.svg", "Total utility vs popular-file importance (Fig. 5c)", "Popular file importance", "Total utility", parseValues(importanceText), true, "utility",
			func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
				return core.GenerateSweepFiles(c, 20.0, 500.0, v), c
			}},
	}

	sweepResultsMap := make(map[string][]lineSeries)

	for _, spec := range specs {
		if len(spec.values) == 0 {
			continue
		}
		series := make([]lineSeries, len(results))
		for i, result := range results {
			series[i] = lineSeries{
				name: result.name,
				x:    spec.values,
				y:    make([]float64, len(spec.values)),
			}
		}

		for point, value := range spec.values {
			pointFiles, pointCfg := spec.configure(cfg, files, value)
			events := generateRequestEvents(pointCfg, seed, requests)
			pointResults := runAllWithConcurrency(pointCfg, pointFiles, events, maxWindow(requests), interval, concurrent)

			for i, result := range pointResults {
				switch spec.metric {
				case "utility":
					series[i].y[point] = result.totalUtility
				case "evictions":
					series[i].y[point] = float64(result.stats.Evictions)
				case "byte_hit_rate":
					series[i].y[point] = result.stats.ByteHitRate() * 100.0
				default:
					if spec.isUtilityMetric {
						series[i].y[point] = result.totalUtility
					} else {
						series[i].y[point] = float64(result.stats.Hits)
					}
				}
			}
		}

		sweepResultsMap[spec.file] = series

		var chartSeries []lineSeries
		for _, s := range series {
			if s.name == "AdaLife-Pro" || s.name == "SMDP-DDQL" || s.name == "Paper-Proposed" {
				dispName := s.name
				if s.name == "SMDP-DDQL" || s.name == "AdaLife-Pro" {
					dispName = "AdaLife-Pro (Ours)"
				} else if s.name == "Paper-Proposed" {
					dispName = "Paper (Niknia '24 DDQL)"
				}
				chartSeries = append(chartSeries, lineSeries{
					name: dispName,
					x:    s.x,
					y:    s.y,
				})
			}
		}

		if err := writeLineChart(filepath.Join(path, spec.file), spec.title, spec.xLabel, spec.yLabel, chartSeries); err != nil {
			return err
		}
	}

	if err := writeSweepCSV(path, specs); err != nil {
		return err
	}

	return writeSweepsJSON(path, specs, sweepResultsMap)
}

func maxWindow(requests int) int {
	window := requests / 10
	if window < 1 {
		return 1
	}
	return window
}

func parseValues(text string) []float64 {
	var values []float64
	for _, item := range strings.Split(text, ",") {
		value, err := strconv.ParseFloat(strings.TrimSpace(item), 64)
		if err == nil && value > 0 {
			values = append(values, value)
		}
	}
	return values
}

func convergenceSeries(results []runResult, interval int) []lineSeries {
	series := make([]lineSeries, len(results))
	for i, result := range results {
		x := result.cumulativeTrials
		if len(x) == 0 {
			x = make([]float64, len(result.cumulativeMisses))
			for j := range x {
				x[j] = float64((j + 1) * interval)
			}
		}
		x, y := downsampleSeries(x, result.cumulativeMisses, 250)
		series[i] = lineSeries{result.name, x, y}
	}
	return series
}

func downsampleSeries(x, y []float64, limit int) ([]float64, []float64) {
	if len(x) <= limit {
		return x, y
	}
	step := float64(len(x)-1) / float64(limit-1)
	downsampledX := make([]float64, 0, limit)
	downsampledY := make([]float64, 0, limit)
	for index := 0; index < limit; index++ {
		position := int(math.Round(float64(index) * step))
		downsampledX = append(downsampledX, x[position])
		downsampledY = append(downsampledY, y[position])
	}
	return downsampledX, downsampledY
}

func writeElasticCapacityArtifacts(path string, results []runResult, interval int, baseCap float64) error {
	var targetResult *runResult
	var paperResult *runResult
	for i := range results {
		if results[i].name == "AdaLife-Pro" || results[i].name == "SMDP-DDQL" {
			targetResult = &results[i]
		} else if results[i].name == "Paper-Proposed" {
			paperResult = &results[i]
		}
	}
	if targetResult == nil || len(targetResult.capacityTrials) == 0 {
		return nil
	}

	x := targetResult.capacityTrials
	effY := targetResult.effectiveCapacities
	usedY := targetResult.usedCapacities

	// Build Base Capacity static line
	baseY := make([]float64, len(x))
	for i := range baseY {
		baseY[i] = baseCap
	}

	xEff, yEff := downsampleSeries(x, effY, 250)
	xUsed, yUsed := downsampleSeries(x, usedY, 250)
	xBase, yBase := downsampleSeries(x, baseY, 250)

	chartSeries := []lineSeries{
		{
			name: "AdaLife-Pro Dynamic Elastic Capacity M_eff(t)",
			x:    xEff,
			y:    yEff,
		},
		{
			name: "AdaLife-Pro Cache Used Memory",
			x:    xUsed,
			y:    yUsed,
		},
	}

	if paperResult != nil && len(paperResult.usedCapacities) > 0 {
		xPaper, yPaper := downsampleSeries(paperResult.capacityTrials, paperResult.usedCapacities, 250)
		chartSeries = append(chartSeries, lineSeries{
			name: "Paper (Niknia '24 DDQL) Used Memory",
			x:    xPaper,
			y:    yPaper,
		})
	}

	chartSeries = append(chartSeries, lineSeries{
		name: "Base Static Capacity M (Rigid Baseline)",
		x:    xBase,
		y:    yBase,
	})

	// 1. Write SVG line chart
	svgFile := filepath.Join(path, "elastic_capacity_vs_requests.svg")
	if err := writeLineChart(svgFile, "Elastic Cache Capacity vs Requests (M_eff dynamic scaling)", "Requests", "Capacity (MiB)", chartSeries); err != nil {
		return err
	}

	// 2. Write CSV trajectory
	csvFile := filepath.Join(path, "elastic_capacity_trajectory.csv")
	file, err := os.Create(csvFile)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	writer.Write([]string{"request", "effective_capacity_mib", "used_capacity_mib", "base_capacity_mib", "headroom_mib"})
	for i := range x {
		eff := effY[i]
		used := usedY[i]
		headroom := eff - baseCap
		writer.Write([]string{
			strconv.FormatFloat(x[i], 'f', 0, 64),
			strconv.FormatFloat(eff, 'f', 2, 64),
			strconv.FormatFloat(used, 'f', 2, 64),
			strconv.FormatFloat(baseCap, 'f', 2, 64),
			strconv.FormatFloat(headroom, 'f', 2, 64),
		})
	}
	writer.Flush()
	if err := writer.Error(); err != nil {
		return err
	}

	// Also copy to data/results folder if present
	dataResultsDir := filepath.Join(path, "..", "..", "data", "results")
	if _, err := os.Stat(filepath.Dir(dataResultsDir)); err == nil {
		_ = os.MkdirAll(dataResultsDir, 0755)
		_ = copyFile(svgFile, filepath.Join(dataResultsDir, "elastic_capacity_vs_requests.svg"))
		_ = copyFile(csvFile, filepath.Join(dataResultsDir, "elastic_capacity_trajectory.csv"))
	}

	return nil
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0644)
}

func mostPopularFile(events []requestEvent) int {
	counts := make(map[int]int)
	popular, highest := 0, 0
	for _, event := range events {
		counts[event.fileID]++
		if counts[event.fileID] > highest {
			popular, highest = event.fileID, counts[event.fileID]
		}
	}
	return popular
}

func modifyPopularFileField(files []core.FileMetadata, value float64, field string, popular int) []core.FileMetadata {
	copyFiles := append([]core.FileMetadata(nil), files...)
	if popular < 0 || popular >= len(copyFiles) {
		return copyFiles
	}
	switch field {
	case "size":
		copyFiles[popular].Size = value
	case "lifetime":
		copyFiles[popular].Lifetime = value
	case "importance":
		copyFiles[popular].Importance = value
	}
	return copyFiles
}

func writeResultsCSV(path string, seed int64, requests, fileCount int, capacity, eta float64, results []runResult) error {
	file, err := os.Create(filepath.Join(path, "results.csv"))
	if err != nil {
		return err
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	writer.Write([]string{
		"seed", "requests", "files", "capacity_mib", "zipf_eta",
		"algorithm", "hit_count", "hit_rate_pct", "byte_hit_rate_pct",
		"total_utility", "warmup_miss_rate_pct", "final_miss_rate_pct",
	})
	for _, r := range results {
		writer.Write([]string{
			fmt.Sprint(seed),
			fmt.Sprint(requests),
			fmt.Sprint(fileCount),
			fmt.Sprintf("%.2f", capacity),
			fmt.Sprintf("%.3f", eta),
			r.name,
			fmt.Sprint(r.stats.Hits),
			fmt.Sprintf("%.4f", r.stats.HitRate()*100),
			fmt.Sprintf("%.4f", r.stats.ByteHitRate()*100),
			fmt.Sprintf("%.6f", r.totalUtility),
			fmt.Sprintf("%.4f", r.warmupMissRate*100),
			fmt.Sprintf("%.4f", r.steadyMissRate*100),
		})
	}
	writer.Flush()
	return writer.Error()
}

func writeResultsJSON(
	path string,
	seed int64,
	requests, fileCount int,
	capacity, eta, lambda float64,
	results []runResult,
) error {
	type algoJSON struct {
		Name                string    `json:"name"`
		Requests            int       `json:"requests"`
		Hits                int       `json:"hits"`
		Misses              int       `json:"misses"`
		HitRatePct          float64   `json:"hit_rate_pct"`
		ByteHitRatePct      float64   `json:"byte_hit_rate_pct"`
		MissRatePct         float64   `json:"miss_rate_pct"`
		Evictions           int       `json:"evictions"`
		EvictionRatePct     float64   `json:"eviction_rate_pct"`
		Rejections          int       `json:"rejections"`
		Insertions          int       `json:"insertions"`
		CachedFiles         int       `json:"cached_files"`
		CapacityMiB         float64   `json:"capacity_mib"`
		UsedCapacityMiB     float64   `json:"used_capacity_mib"`
		UtilizationPct      float64   `json:"utilization_pct"`
		TotalUtility        float64   `json:"total_utility"`
		WarmupMissRatePct   float64   `json:"warmup_miss_rate_pct"`
		FinalMissRatePct    float64   `json:"final_miss_rate_pct"`
		CumulativeTrials    []float64 `json:"cumulative_trials"`
		CumulativeMissesPct []float64 `json:"cumulative_misses_pct"`
	}

	type payload struct {
		Metadata struct {
			Timestamp   string  `json:"timestamp"`
			Seed        int64   `json:"seed"`
			Requests    int     `json:"requests"`
			Files       int     `json:"files"`
			CapacityMiB float64 `json:"capacity_mib"`
			ZipfEta     float64 `json:"zipf_eta"`
			Lambda      float64 `json:"lambda_source"`
		} `json:"metadata"`
		Algorithms []algoJSON `json:"algorithms"`
	}

	var p payload
	p.Metadata.Timestamp = time.Now().UTC().Format(time.RFC3339)
	p.Metadata.Seed = seed
	p.Metadata.Requests = requests
	p.Metadata.Files = fileCount
	p.Metadata.CapacityMiB = capacity
	p.Metadata.ZipfEta = eta
	p.Metadata.Lambda = lambda

	for _, r := range results {
		p.Algorithms = append(p.Algorithms, algoJSON{
			Name:                r.name,
			Requests:            r.stats.Requests,
			Hits:                r.stats.Hits,
			Misses:              r.stats.Misses,
			HitRatePct:          r.stats.HitRate() * 100.0,
			ByteHitRatePct:      r.stats.ByteHitRate() * 100.0,
			MissRatePct:         r.stats.MissRate() * 100.0,
			Evictions:           r.stats.Evictions,
			EvictionRatePct:     r.stats.EvictionRate() * 100.0,
			Rejections:          r.stats.RejectedRequests,
			Insertions:          r.stats.Insertions,
			CachedFiles:         r.stats.CachedFiles,
			CapacityMiB:         r.stats.Capacity,
			UsedCapacityMiB:     r.stats.UsedCapacity,
			UtilizationPct:      r.stats.Utilization() * 100.0,
			TotalUtility:        r.totalUtility,
			WarmupMissRatePct:   r.warmupMissRate * 100.0,
			FinalMissRatePct:    r.steadyMissRate * 100.0,
			CumulativeTrials:    r.cumulativeTrials,
			CumulativeMissesPct: r.cumulativeMisses,
		})
	}

	bytes, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}

	// Write to graphs folder
	if err := os.WriteFile(filepath.Join(path, "baseline_results.json"), bytes, 0644); err != nil {
		return err
	}

	// Also write to data/results folder if present
	dataResultsDir := filepath.Join(path, "..", "..", "data", "results")
	if _, err := os.Stat(filepath.Dir(dataResultsDir)); err == nil {
		_ = os.MkdirAll(dataResultsDir, 0755)
		_ = os.WriteFile(filepath.Join(dataResultsDir, "baseline_results.json"), bytes, 0644)
	}

	return nil
}

func writeSweepsJSON(path string, specs []sweepSpec, seriesMap map[string][]lineSeries) error {
	type seriesJSON struct {
		Algorithm string    `json:"algorithm"`
		YValues   []float64 `json:"y_values"`
	}

	type sweepJSON struct {
		File        string       `json:"file"`
		Title       string       `json:"title"`
		XParameter  string       `json:"x_parameter"`
		YMetric     string       `json:"y_metric"`
		XValues     []float64    `json:"x_values"`
		IsUtility   bool         `json:"is_utility_metric"`
		Series      []seriesJSON `json:"series"`
	}

	type payload struct {
		Timestamp string      `json:"timestamp"`
		Sweeps    []sweepJSON `json:"sweeps"`
	}

	var p payload
	p.Timestamp = time.Now().UTC().Format(time.RFC3339)

	for _, spec := range specs {
		seriesList, exists := seriesMap[spec.file]
		if !exists {
			continue
		}
		var sJSON []seriesJSON
		for _, s := range seriesList {
			sJSON = append(sJSON, seriesJSON{
				Algorithm: s.name,
				YValues:   s.y,
			})
		}
		p.Sweeps = append(p.Sweeps, sweepJSON{
			File:       spec.file,
			Title:      spec.title,
			XParameter: spec.xLabel,
			YMetric:    spec.yLabel,
			XValues:    spec.values,
			IsUtility:  spec.isUtilityMetric,
			Series:     sJSON,
		})
	}

	bytes, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(filepath.Join(path, "parameter_sweeps.json"), bytes, 0644); err != nil {
		return err
	}

	dataResultsDir := filepath.Join(path, "..", "..", "data", "results")
	if _, err := os.Stat(filepath.Dir(dataResultsDir)); err == nil {
		_ = os.MkdirAll(dataResultsDir, 0755)
		_ = os.WriteFile(filepath.Join(dataResultsDir, "parameter_sweeps.json"), bytes, 0644)
	}

	return nil
}

func writeMDPComparisonJSON(path string, mdpResults []baselines.MDPEvaluationResult) error {
	type payload struct {
		Timestamp string                          `json:"timestamp"`
		Title     string                          `json:"title"`
		Results   []baselines.MDPEvaluationResult `json:"results"`
	}
	p := payload{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Title:     "Table III: Hit rate for MDP and SMDP under different time quotas and request rates",
		Results:   mdpResults,
	}

	bytes, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}

	_ = os.WriteFile(filepath.Join(path, "mdp_vs_smdp_comparison.json"), bytes, 0644)

	dataResultsDir := filepath.Join(path, "..", "..", "data", "results")
	if _, err := os.Stat(filepath.Dir(dataResultsDir)); err == nil {
		_ = os.MkdirAll(dataResultsDir, 0755)
		_ = os.WriteFile(filepath.Join(dataResultsDir, "mdp_vs_smdp_comparison.json"), bytes, 0644)
	}

	return nil
}

func writeSweepCSV(path string, specs []sweepSpec) error {
	file, err := os.Create(filepath.Join(path, "sweep_parameters.csv"))
	if err != nil {
		return err
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	writer.Write([]string{"chart", "x_values"})
	for _, spec := range specs {
		values := make([]string, len(spec.values))
		for i, v := range spec.values {
			values[i] = strconv.FormatFloat(v, 'g', -1, 64)
		}
		writer.Write([]string{spec.file, strings.Join(values, "|")})
	}
	writer.Flush()
	return writer.Error()
}

func writeLineChart(path, title, xLabel, yLabel string, series []lineSeries) error {
	const width, height, left, top, chartWidth, chartHeight = 1200, 700, 100, 80, 1000, 480
	minX, maxX, maxY := math.Inf(1), math.Inf(-1), 1.0
	for _, line := range series {
		for _, value := range line.x {
			minX = math.Min(minX, value)
			maxX = math.Max(maxX, value)
		}
		for _, value := range line.y {
			maxY = math.Max(maxY, value)
		}
	}
	if math.IsInf(minX, 1) {
		minX = 0
	}
	if minX == maxX {
		minX -= 1
		maxX += 1
	}
	xPadding := (maxX - minX) * 0.05
	minX -= xPadding
	maxX += xPadding
	maxY *= 1.15
	if maxY <= 0 {
		maxY = 1.0
	}

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	colors := []string{"#006d77", "#c44536", "#386641", "#6a4c93", "#e76f51", "#2a9d8f"}
	dashes := []string{"", "10 6", "3 5", "14 5 3 5", "6 4 2 4", "8 4"}

	fmt.Fprintf(file, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n", width, height, width, height)
	fmt.Fprintln(file, "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#243447}.title{font-size:25px;font-weight:600}.label{font-size:15px}.legend{font-size:14px;font-weight:600}.grid{stroke:#d9e2ec}.axis{stroke:#718096}.line{fill:none;stroke-width:3}.point{stroke:#f7fafc;stroke-width:2}</style>")
	fmt.Fprintf(file, "<rect width=\"100%%\" height=\"100%%\" fill=\"#f7fafc\"/><text x=\"%d\" y=\"42\" class=\"title\">%s</text><text x=\"%d\" y=\"%d\" class=\"label\" text-anchor=\"middle\">%s</text><text x=\"20\" y=\"%d\" class=\"label\" transform=\"rotate(-90 20 %d)\">%s</text>\n",
		left, title, left+chartWidth/2, top+chartHeight+70, xLabel, top+chartHeight/2, top+chartHeight/2, yLabel)

	for grid := 0; grid <= 5; grid++ {
		y := top + chartHeight - grid*chartHeight/5
		value := maxY * float64(grid) / 5
		fmt.Fprintf(file, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"grid\"/><text x=\"%d\" y=\"%d\" class=\"label\" text-anchor=\"end\">%.2f</text>\n",
			left, y, left+chartWidth, y, left-10, y+5, value)
	}

	if len(series) > 0 && len(series[0].x) > 0 {
		for tick, value := range series[0].x {
			x := float64(left) + (value-minX)/(maxX-minX)*chartWidth
			fmt.Fprintf(file, "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" class=\"grid\"/><text x=\"%.1f\" y=\"%d\" class=\"label\" text-anchor=\"middle\">%.3g</text>\n",
				x, top, x, top+chartHeight, x, top+chartHeight+25, value)
			if tick == 7 {
				break
			}
		}
	}

	for i, line := range series {
		if len(line.x) == 0 {
			continue
		}
		points := make([]string, len(line.x))
		for j := range line.x {
			x := float64(left) + (line.x[j]-minX)/(maxX-minX)*chartWidth
			y := float64(top+chartHeight) - line.y[j]/maxY*chartHeight
			points[j] = fmt.Sprintf("%.1f,%.1f", x, y)
		}
		legendY := 24 + i*22
		fmt.Fprintf(file, "<polyline points=\"%s\" class=\"line\" stroke=\"%s\" stroke-dasharray=\"%s\"/><line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"4\" stroke-dasharray=\"%s\"/><text x=\"%d\" y=\"%d\" class=\"legend\">%s</text>",
			strings.Join(points, " "), colors[i%len(colors)], dashes[i%len(dashes)],
			width-290, legendY-5, width-260, legendY-5, colors[i%len(colors)], dashes[i%len(dashes)], width-250, legendY, line.name)
		for j := range line.x {
			x := float64(left) + (line.x[j]-minX)/(maxX-minX)*chartWidth
			y := float64(top+chartHeight) - line.y[j]/maxY*chartHeight
			fmt.Fprintf(file, "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"5\" fill=\"%s\" class=\"point\"/>", x, y, colors[i%len(colors)])
		}
		fmt.Fprintln(file)
	}

	fmt.Fprintf(file, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"axis\"/><line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"axis\"/></svg>\n",
		left, top, left, top+chartHeight, left, top+chartHeight, left+chartWidth, top+chartHeight)
	return nil
}
