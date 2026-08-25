package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

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
	configure                   func(*config.Config, []core.FileMetadata, float64) ([]core.FileMetadata, *config.Config)
}

func prepareGraphsDirectory(path string) error {
	if err := os.RemoveAll(path); err != nil {
		return err
	}
	return os.MkdirAll(path, 0755)
}

func writeGraphs(path string, seed int64, requests, fileCount int, capacity, eta float64, interval int, cacheSizeText, rateText, etaText, lifetimeText, sizeText string, cfg *config.Config, files []core.FileMetadata, events []requestEvent, results []runResult, concurrent bool) error {
	if err := writeResultsCSV(path, seed, requests, fileCount, capacity, eta, results); err != nil {
		return err
	}
	if err := writeLineChart(filepath.Join(path, "cumulative_hit_rate.svg"), "Hit-rate convergence", "Trial", "Cumulative hit rate (%)", convergenceSeries(results, interval)); err != nil {
		return err
	}
	popularFile := mostPopularFile(events)
	base := []sweepSpec{
		{"hit_rate_vs_cache_size.svg", "Hit rate vs cache size", "Cache size (MiB)", "Hit rate (%)", parseValues(cacheSizeText), func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
			n := *c
			n.CacheCapacity = v
			return f, &n
		}},
		{"hit_rate_vs_request_rate.svg", "Hit rate vs request rate", "Request rate lambda", "Hit rate (%)", parseValues(rateText), func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
			n := *c
			n.LambdaSource = v
			return f, &n
		}},
		{"hit_rate_vs_zipf_eta.svg", "Hit rate vs popularity skewness", "Zipf eta", "Hit rate (%)", parseValues(etaText), func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
			n := *c
			n.ZipfEta = v
			return f, &n
		}},
		{"hit_rate_vs_file_lifetime.svg", "Hit rate vs popular-file lifetime", "File lifetime", "Hit rate (%)", parseValues(lifetimeText), func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
			return modifyPopularFile(f, v, false, popularFile), c
		}},
		{"hit_rate_vs_file_size.svg", "Hit rate vs popular-file size", "File size (MiB)", "Hit rate (%)", parseValues(sizeText), func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
			return modifyPopularFile(f, v, true, popularFile), c
		}},
		{"total_utility_vs_cache_size.svg", "Total utility vs cache size", "Cache size (MiB)", "Total utility", parseValues(cacheSizeText), func(c *config.Config, f []core.FileMetadata, v float64) ([]core.FileMetadata, *config.Config) {
			n := *c
			n.CacheCapacity = v
			return f, &n
		}},
	}
	for _, spec := range base {
		series := make([]lineSeries, len(results))
		for i, result := range results {
			series[i] = lineSeries{name: result.name, x: spec.values, y: make([]float64, len(spec.values))}
		}
		for point, value := range spec.values {
			var pointFiles []core.FileMetadata
			pointFiles, pointCfg := spec.configure(cfg, files, value)
			events := generateRequestEvents(pointCfg, seed, requests)
			pointResults := runAllWithConcurrency(pointCfg, pointFiles, events, maxWindow(requests), interval, concurrent)
			for i, result := range pointResults {
				if spec.file == "total_utility_vs_cache_size.svg" {
					series[i].y[point] = result.totalUtility
				} else {
					series[i].y[point] = result.stats.HitRate() * 100
				}
			}
		}
		if err := writeLineChart(filepath.Join(path, spec.file), spec.title, spec.xLabel, spec.yLabel, series); err != nil {
			return err
		}
	}
	return writeSweepCSV(path, base)
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
			x = make([]float64, len(result.cumulativeHits))
			for j := range x {
				x[j] = float64((j + 1) * interval)
			}
		}
		x, y := downsampleSeries(x, result.cumulativeHits, 250)
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

func modifyPopularFile(files []core.FileMetadata, value float64, size bool, popular int) []core.FileMetadata {
	copyFiles := append([]core.FileMetadata(nil), files...)
	if popular < 0 || popular >= len(copyFiles) {
		return copyFiles
	}
	if size {
		copyFiles[popular].Size = value
	} else {
		copyFiles[popular].Lifetime = value
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
	writer.Write([]string{"seed", "requests", "files", "capacity_mib", "zipf_eta", "algorithm", "hit_count", "hit_rate_pct", "byte_hit_rate_pct", "total_utility", "warmup_miss_rate_pct", "final_miss_rate_pct"})
	for _, r := range results {
		writer.Write([]string{fmt.Sprint(seed), fmt.Sprint(requests), fmt.Sprint(fileCount), fmt.Sprintf("%.2f", capacity), fmt.Sprintf("%.3f", eta), r.name, fmt.Sprint(r.stats.Hits), fmt.Sprintf("%.4f", r.stats.HitRate()*100), fmt.Sprintf("%.4f", r.stats.ByteHitRate()*100), fmt.Sprintf("%.6f", r.totalUtility), fmt.Sprintf("%.4f", r.warmupMissRate*100), fmt.Sprintf("%.4f", r.steadyMissRate*100)})
	}
	writer.Flush()
	return writer.Error()
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
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	colors := []string{"#006d77", "#c44536", "#386641", "#6a4c93"}
	dashes := []string{"", "10 6", "3 5", "14 5 3 5"}
	fmt.Fprintf(file, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n", width, height, width, height)
	fmt.Fprintln(file, "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#243447}.title{font-size:25px;font-weight:600}.label{font-size:15px}.legend{font-size:14px;font-weight:600}.grid{stroke:#d9e2ec}.axis{stroke:#718096}.line{fill:none;stroke-width:3}.point{stroke:#f7fafc;stroke-width:2}</style>")
	fmt.Fprintf(file, "<rect width=\"100%%\" height=\"100%%\" fill=\"#f7fafc\"/><text x=\"%d\" y=\"42\" class=\"title\">%s</text><text x=\"%d\" y=\"%d\" class=\"label\" text-anchor=\"middle\">%s</text><text x=\"20\" y=\"%d\" class=\"label\" transform=\"rotate(-90 20 %d)\">%s</text>\n", left, title, left+chartWidth/2, top+chartHeight+70, xLabel, top+chartHeight/2, top+chartHeight/2, yLabel)
	for grid := 0; grid <= 5; grid++ {
		y := top + chartHeight - grid*chartHeight/5
		value := maxY * float64(grid) / 5
		fmt.Fprintf(file, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"grid\"/><text x=\"%d\" y=\"%d\" class=\"label\" text-anchor=\"end\">%.2f</text>\n", left, y, left+chartWidth, y, left-10, y+5, value)
	}
	if len(series) > 0 {
		for tick, value := range series[0].x {
			x := float64(left) + (value-minX)/(maxX-minX)*chartWidth
			fmt.Fprintf(file, "<line x1=\"%.1f\" y1=\"%d\" x2=\"%.1f\" y2=\"%d\" class=\"grid\"/><text x=\"%.1f\" y=\"%d\" class=\"label\" text-anchor=\"middle\">%.3g</text>\n", x, top, x, top+chartHeight, x, top+chartHeight+25, value)
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
		fmt.Fprintf(file, "<polyline points=\"%s\" class=\"line\" stroke=\"%s\" stroke-dasharray=\"%s\"/><line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" stroke=\"%s\" stroke-width=\"4\" stroke-dasharray=\"%s\"/><text x=\"%d\" y=\"%d\" class=\"legend\">%s</text>", strings.Join(points, " "), colors[i%len(colors)], dashes[i%len(dashes)], width-290, legendY-5, width-260, legendY-5, colors[i%len(colors)], dashes[i%len(dashes)], width-250, legendY, line.name)
		for j := range line.x {
			x := float64(left) + (line.x[j]-minX)/(maxX-minX)*chartWidth
			y := float64(top+chartHeight) - line.y[j]/maxY*chartHeight
			fmt.Fprintf(file, "<circle cx=\"%.1f\" cy=\"%.1f\" r=\"5\" fill=\"%s\" class=\"point\"/>", x, y, colors[i%len(colors)])
		}
		fmt.Fprintln(file)
	}
	fmt.Fprintf(file, "<line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"axis\"/><line x1=\"%d\" y1=\"%d\" x2=\"%d\" y2=\"%d\" class=\"axis\"/></svg>\n", left, top, left, top+chartHeight, left, top+chartHeight, left+chartWidth, top+chartHeight)
	return nil
}
