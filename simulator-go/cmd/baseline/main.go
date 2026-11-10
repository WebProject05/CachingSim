package main

import (
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
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

type timeAwareCache interface {
	AccessAt(int, float64) bool
	Stats() baselines.CacheStats
}

type utilityAwareCache interface {
	timeAwareCache
	LastHitUtility() float64
}

// smdpCacheWrapper adapts core.CacheEngine to the cache interface.
type smdpCacheWrapper struct {
	engine         *core.CacheEngine
	files          []core.FileMetadata
	cfg            *config.Config
	requests       int
	hits           int
	evictions      int
	rejected       int
	insertions     int
	requestedBytes float64
	hitBytes       float64
	missBytes      float64
	lastTime       float64
	hasStarted     bool
	lastHitUtil    float64
}

func newSMDPCacheWrapper(cfg *config.Config, files []core.FileMetadata) *smdpCacheWrapper {
	return &smdpCacheWrapper{
		engine: core.NewCacheEngine(cfg, files),
		files:  core.CloneFiles(files),
		cfg:    cfg,
	}
}

func (w *smdpCacheWrapper) AccessAt(fileID int, currentTime float64) bool {
	w.requests++
	w.lastHitUtil = 0.0
	if fileID < 0 || fileID >= len(w.engine.Files) {
		w.rejected++
		return false
	}
	fileSize := w.engine.Files[fileID].Size
	if fileSize > 0 {
		w.requestedBytes += fileSize
	}

	if w.hasStarted && currentTime > w.lastTime {
		w.engine.UpdateTrafficRate(currentTime - w.lastTime)
	} else if !w.hasStarted && currentTime > 0 {
		w.engine.UpdateTrafficRate(currentTime)
	}
	w.lastTime = currentTime
	w.hasStarted = true

	w.engine.RecordRequest(fileID)

	action := 1
	if !w.engine.IsFresh(fileID, currentTime) {
		if !w.shouldAdmit(fileID, currentTime) {
			action = 0
		}
	}

	isHit, utilGained, evicts, inserted := w.engine.ProcessAdaLifeStep(fileID, action, currentTime)
	w.evictions += evicts

	if isHit {
		w.hits++
		w.lastHitUtil = utilGained
		if fileSize > 0 {
			w.hitBytes += fileSize
		}
		return true
	}

	if fileSize > 0 {
		w.missBytes += fileSize
	}
	if fileSize > w.engine.EffectiveCapacity() {
		w.rejected++
		return false
	}
	if inserted {
		w.insertions++
	}
	return false
}

func (w *smdpCacheWrapper) LastHitUtility() float64 {
	return w.lastHitUtil
}

func (w *smdpCacheWrapper) shouldAdmit(fileID int, currentTime float64) bool {
	fileSize := w.engine.Files[fileID].Size
	effCap := w.engine.EffectiveCapacity()
	if fileSize > effCap {
		return false
	}

	popVec := w.engine.GetPopularityVector()
	candPop := 0.5
	if fileID < len(popVec) {
		candPop += popVec[fileID]
	}
	candUtil := w.cfg.UTMax * w.engine.Files[fileID].Importance
	candCost := w.engine.Files[fileID].FetchCost(w.cfg)
	candValue := candUtil * candCost

	type cachedItem struct {
		id          int
		u           float64
		size        float64
		pop         float64
		density     float64
		inProbation bool
	}

	var protectedItems []cachedItem
	var probationItems []cachedItem
	for id, c := range w.engine.Cached {
		if !c || w.engine.Files[id].Freshness(currentTime) >= 1.0 {
			continue
		}
		p := 0.5
		if id < len(popVec) {
			p += popVec[id]
		}
		sz := math.Max(1.0, w.engine.Files[id].Size)
		u := w.engine.Files[id].Utility(currentTime, w.cfg) * w.engine.Files[id].FetchCost(w.cfg)
		inProb := id < len(w.engine.InProbation) && w.engine.InProbation[id]
		item := cachedItem{
			id:          id,
			u:           u,
			size:        sz,
			pop:         p,
			density:     u / sz,
			inProbation: inProb,
		}
		if inProb {
			probationItems = append(probationItems, item)
		} else {
			protectedItems = append(protectedItems, item)
		}
	}

	if w.cfg.EnableProbationShield {
		isGhostHit := w.engine.PeekGhost(fileID, currentTime)
		isReturningPopular := isGhostHit || candPop >= 2.0
		probCap, protCap := w.engine.TierCapacities(fileSize)

		// Repeat arrival seen in Ghost Cache or popular in window -> evaluate admission into Protected Main Tier
		if isReturningPopular && protCap >= fileSize {
			freeProt := protCap - w.engine.ProtectedUsed
			if freeProt >= fileSize {
				return true
			}
			needed := fileSize - freeProt
			sort.Slice(protectedItems, func(i, j int) bool {
				return protectedItems[i].density < protectedItems[j].density
			})
			var freed float64
			var victimTotalScore float64
			for _, item := range protectedItems {
				if freed >= needed {
					break
				}
				freed += item.size
				victimTotalScore += (item.pop * item.u) * (item.size / math.Max(1.0, fileSize))
			}
			// Returning ghost item receives repeat-request priority boost
			return (math.Max(1.5, candPop) * candValue) >= (victimTotalScore * 0.85)
		}

		// 1st arrival: allow ProcessAdaLifeStep to use free cache space or Probation tier
		probMaxBudget := math.Max(probCap, effCap-w.engine.ProtectedUsed)
		if fileSize > probMaxBudget {
			return true
		}
		freeTotal := effCap - w.engine.UsedCapacity
		if freeTotal >= fileSize {
			return true
		}

		// Cache is full: admit into Probation unless incoming candidate is far weaker than Probation victims
		neededProb := fileSize - freeTotal
		sort.Slice(probationItems, func(i, j int) bool {
			return probationItems[i].density < probationItems[j].density
		})
		var freed float64
		var victimTotalScore float64
		for _, item := range probationItems {
			if freed >= neededProb {
				break
			}
			freed += item.size
			victimTotalScore += (item.pop * item.u) * (item.size / math.Max(1.0, fileSize))
		}
		if victimTotalScore <= 0 {
			return true
		}
		return (candPop * candValue) >= (victimTotalScore * 0.50)
	}

	// Single-tier mode (Part 1 + enabled cost/elastic features)
	freeSpace := effCap - w.engine.UsedCapacity
	if freeSpace >= fileSize {
		return true
	}

	needed := fileSize - freeSpace
	if w.cfg.EvictionStrategy != "lowest_utility" {
		sort.Slice(protectedItems, func(i, j int) bool {
			return protectedItems[i].density < protectedItems[j].density
		})
		var freed float64
		var victimTotalScore float64
		for _, item := range protectedItems {
			if freed >= needed {
				break
			}
			freed += item.size
			victimTotalScore += (item.pop * item.u) * (item.size / math.Max(1.0, fileSize))
		}
		candTotalScore := candPop * candValue
		return candTotalScore >= victimTotalScore
	}

	// Legacy lowest-utility sorting
	sort.Slice(protectedItems, func(i, j int) bool {
		return protectedItems[i].u < protectedItems[j].u
	})
	candScore := (math.Pow(candPop, 1.25) * candValue) / math.Pow(fileSize, 0.25)
	var freed float64
	var victimValSum float64
	var victimCount float64
	for _, item := range protectedItems {
		if freed >= needed {
			break
		}
		freed += item.size
		victimValSum += (math.Pow(item.pop, 1.25) * item.u) / math.Pow(item.size, 0.25)
		victimCount += 1.0
	}
	if victimCount <= 0 {
		return true
	}
	victimScore := victimValSum / victimCount
	return candScore >= (victimScore * 0.70)
}

func (w *smdpCacheWrapper) Access(fileID int) bool {
	return w.AccessAt(fileID, w.engine.CurrentTime)
}

func (w *smdpCacheWrapper) Stats() baselines.CacheStats {
	cachedCount := 0
	for _, c := range w.engine.Cached {
		if c {
			cachedCount++
		}
	}
	return baselines.CacheStats{
		Requests:         w.requests,
		Hits:             w.hits,
		Misses:           w.requests - w.hits,
		Evictions:        w.evictions,
		RejectedRequests: w.rejected,
		Insertions:       w.insertions,
		CachedFiles:      cachedCount,
		Capacity:         w.engine.EffectiveCapacity(),
		UsedCapacity:     w.engine.UsedCapacity,
		RequestedBytes:   w.requestedBytes,
		HitBytes:         w.hitBytes,
		MissBytes:        w.missBytes,
	}
}

// ctdCacheWrapper adapts CTDCache to the cache interface with timestamps.
type ctdCacheWrapper struct {
	ctd         *baselines.CTDCache
	lastHitUtil float64
}

func newCTDCacheWrapper(cfg *config.Config, files []core.FileMetadata) *ctdCacheWrapper {
	return &ctdCacheWrapper{
		ctd: baselines.NewCTDCache(cfg, files),
	}
}

func (w *ctdCacheWrapper) AccessAt(fileID int, currentTime float64) bool {
	prevUtil := w.ctd.TotalUtility()
	hit := w.ctd.AccessAtTime(fileID, currentTime)
	if hit {
		w.lastHitUtil = w.ctd.TotalUtility() - prevUtil
	} else {
		w.lastHitUtil = 0.0
	}
	return hit
}

func (w *ctdCacheWrapper) LastHitUtility() float64 {
	return w.lastHitUtil
}

func (w *ctdCacheWrapper) Access(fileID int) bool {
	return w.ctd.Access(fileID)
}

func (w *ctdCacheWrapper) Stats() baselines.CacheStats {
	return w.ctd.Stats()
}

type runResult struct {
	name                                                         string
	stats                                                        baselines.CacheStats
	elapsedSeconds, warmupMissRate, steadyMissRate, totalUtility float64
	cumulativeTrials, cumulativeMisses                           []float64
	capacityTrials, effectiveCapacities, usedCapacities          []float64
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
	generateGraphs := flag.Bool("g", false, "generate graphs after the experiment")
	graphInterval := flag.Int("graph-interval", 100, "trials between convergence graph points")
	concurrent := flag.Bool("concurrent", false, "run cache algorithms concurrently; default is sequential")
	cacheSizes := flag.String("cache-sizes", "5000,7500,10000,12500,15000", "cache sizes in MiB")
	requestRates := flag.String("request-rates", "0.5,1.5,2.5,3.5,4.5,5.5,6.5", "lambda request rates")
	zipfEtas := flag.String("zipf-etas", "0.0,0.2,0.4,0.6,0.8,1.0", "Zipf eta values")
	fileLifetimes := flag.String("file-lifetimes", "10,15,20,25,30", "popular-file lifetime values")
	fileSizes := flag.String("file-sizes", "100,200,500,700,1000", "popular-file sizes in MiB")
	fileImportances := flag.String("file-importances", "0.1,0.3,0.5,0.7,0.9", "popular-file importance values")
	runMDPTbl := flag.Bool("mdp-table", false, "run and print Table III (MDP vs SMDP time quota comparison)")
	lambdaRate := flag.Float64("lambda", 0.2, "Poisson request arrival rate lambda (default: 0.2; use 5.0 for high request rate)")
	evictStrategy := flag.String("evict", "utility_density", "eviction strategy: 'utility_density' (resolves size-blindness) or 'lowest_utility' (legacy paper)")
	memPenalty := flag.Float64("mem-penalty", 10.0, "penalty weight for unoccupied memory (default: 10.0 resolving empty-cache trap; 100.0 for legacy paper)")
	enableAdaLife := flag.Bool("adalife", true, "enable AdaLife dynamic lifetime lease extensions")
	enableElastic := flag.Bool("elastic", true, "enable elastic cache capacity headroom during bursts")
	enableProbation := flag.Bool("probation", true, "enable two-tier probationary shield for one-hit wonders")
	enableFetchCost := flag.Bool("fetch-cost", true, "enable asymmetric origin fetch-cost weighting")
	enableGhost := flag.Bool("ghost", true, "enable counterfactual ghost cache regret and fast-track")
	part1Only := flag.Bool("part1-only", false, "disable all Part 2 AdaLife-Pro mechanisms (run Part 1 baseline only)")
	flag.Parse()

	if *generateGraphs {
		if err := prepareGraphsDirectory(*graphsPath); err != nil {
			fmt.Fprintf(os.Stderr, "could not prepare graph directory: %v\n", err)
			return
		}
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
	if *lambdaRate > 0 {
		cfg.LambdaSource = *lambdaRate
	}
	if *evictStrategy != "" {
		cfg.EvictionStrategy = *evictStrategy
	}
	if *memPenalty >= 0 {
		cfg.MemPenaltyWeight = *memPenalty
	}
	if *part1Only {
		cfg.EnableAdaLife = false
		cfg.EnableElasticCapacity = false
		cfg.EnableProbationShield = false
		cfg.EnableFetchCost = false
		cfg.EnableGhostCache = false
	} else {
		cfg.EnableAdaLife = *enableAdaLife
		cfg.EnableElasticCapacity = *enableElastic
		cfg.EnableProbationShield = *enableProbation
		cfg.EnableFetchCost = *enableFetchCost
		cfg.EnableGhostCache = *enableGhost
	}
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

	modeLabel := "AdaLife-Pro (Part 1 + Part 2 Full Engine)"
	if !cfg.EnableAdaLife && !cfg.EnableElasticCapacity && !cfg.EnableProbationShield && !cfg.EnableFetchCost && !cfg.EnableGhostCache {
		modeLabel = "Part 1 Only (AdaLife-Pro Disabled)"
	}

	printBox("SMDP EDGE CACHING EXPERIMENT: PROPOSED (SMDP-DDQL / AdaLife-Pro) vs CTD", []string{
		fmt.Sprintf("Mode              : %s", modeLabel),
		fmt.Sprintf("Features          : AdaLife=%t | Elastic=%t | Probation=%t | FetchCost=%t | Ghost=%t",
			cfg.EnableAdaLife, cfg.EnableElasticCapacity, cfg.EnableProbationShield, cfg.EnableFetchCost, cfg.EnableGhostCache),
		fmt.Sprintf("Seed              : %d", *seed),
		fmt.Sprintf("Requests          : %d", *requests),
		fmt.Sprintf("Files             : %d", *fileCount),
		fmt.Sprintf("Capacity          : %.2f MiB", *capacity),
		fmt.Sprintf("Zipf eta          : %.3f", cfg.ZipfEta),
		fmt.Sprintf("Lambda source     : %.3f", cfg.LambdaSource),
		fmt.Sprintf("Measurement window: %d", measurementWindow),
	})
	printFileStats(files)

	results := runAllWithConcurrency(cfg, files, events, measurementWindow, *graphInterval, *concurrent)
	printComparisonTable(results)
	for _, result := range results {
		if result.name == "AdaLife-Pro" || result.name == "SMDP-DDQL" || result.name == "Paper-Proposed" || result.name == "CTD" {
			printResult(result)
		}
	}

	if *runMDPTbl || *generateGraphs {
		mdpResults := baselines.RunMDPvsSMDPComparison(cfg, files, *seed, 1000, []float64{5.0, 1.66, 1.0}, []float64{0.2, 0.4, 0.6, 0.8, 1.0})
		_ = writeMDPComparisonJSON(*graphsPath, mdpResults)
		if *runMDPTbl {
			printTableIII(mdpResults)
		}
	}

	if *generateGraphs {
		if err := writeGraphs(*graphsPath, *seed, *requests, *fileCount, *capacity, cfg.ZipfEta, *graphInterval,
			*cacheSizes, *requestRates, *zipfEtas, *fileLifetimes, *fileSizes, *fileImportances,
			cfg, files, events, results, *concurrent); err != nil {
			fmt.Fprintf(os.Stderr, "could not write graphs: %v\n", err)
			return
		}
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
	paperCfg := *cfg
	paperCfg.EvictionStrategy = "lowest_utility"
	paperCfg.MemPenaltyWeight = 100.0
	paperCfg.EnableAdaLife = false
	paperCfg.EnableElasticCapacity = false
	paperCfg.EnableProbationShield = false
	paperCfg.EnableFetchCost = false
	paperCfg.EnableGhostCache = false

	proposedName := "AdaLife-Pro"
	if !cfg.EnableAdaLife && !cfg.EnableElasticCapacity && !cfg.EnableProbationShield && !cfg.EnableFetchCost && !cfg.EnableGhostCache {
		proposedName = "SMDP-DDQL"
	}

	algorithms := []struct {
		name     string
		newCache func() cache
	}{
		{"FIFO", func() cache { return baselines.NewFIFOCache(cfg, files) }},
		{"LRU", func() cache { return baselines.NewLRUCache(cfg, files) }},
		{"LFU", func() cache { return baselines.NewLFUCache(cfg, files) }},
		{"SIEVE", func() cache { return baselines.NewSIEVECache(cfg, files) }},
		{"CTD", func() cache { return newCTDCacheWrapper(cfg, files) }},
		{"Paper-Proposed", func() cache { return newSMDPCacheWrapper(&paperCfg, files) }},
		{proposedName, func() cache { return newSMDPCacheWrapper(cfg, files) }},
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
	capacityTrials := make([]float64, 0, cap(cumulative))
	effectiveCapacities := make([]float64, 0, cap(cumulative))
	usedCapacities := make([]float64, 0, cap(cumulative))
	start := time.Now()

	for index, event := range events {
		isHit := false
		if tac, ok := c.(timeAwareCache); ok {
			isHit = tac.AccessAt(event.fileID, event.time)
		} else {
			isHit = c.Access(event.fileID)
		}

		if isHit {
			hits++
			if uac, ok := c.(utilityAwareCache); ok {
				utility += uac.LastHitUtility()
			} else if event.fileID >= 0 && event.fileID < len(files) {
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
			st := c.Stats()
			capacityTrials = append(capacityTrials, float64(index+1))
			effectiveCapacities = append(effectiveCapacities, st.Capacity)
			usedCapacities = append(usedCapacities, st.UsedCapacity)
		}
	}
	return runResult{
		name:                name,
		stats:               c.Stats(),
		elapsedSeconds:      time.Since(start).Seconds(),
		warmupMissRate:      float64(warmup) / float64(window),
		steadyMissRate:      float64(steady) / float64(window),
		totalUtility:        utility,
		cumulativeTrials:    cumulativeTrials,
		cumulativeMisses:    cumulative,
		capacityTrials:      capacityTrials,
		effectiveCapacities: effectiveCapacities,
		usedCapacities:      usedCapacities,
	}
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
	printBox(result.name, []string{
		fmt.Sprintf("Requests          : %d", s.Requests),
		fmt.Sprintf("Hits              : %d", s.Hits),
		fmt.Sprintf("Misses            : %d", s.Misses),
		fmt.Sprintf("Hit rate          : %.2f%%", s.HitRate()*100),
		fmt.Sprintf("Byte hit rate     : %.2f%%", s.ByteHitRate()*100),
		fmt.Sprintf("Evictions         : %d", s.Evictions),
		fmt.Sprintf("Utilization       : %.2f%%", s.Utilization()*100),
		fmt.Sprintf("Total utility     : %.4f", result.totalUtility),
		fmt.Sprintf("Average ns/request: %s", latency),
		fmt.Sprintf("Operations/sec    : %s", ops),
		fmt.Sprintf("Warm-up miss rate : %.2f%%", result.warmupMissRate*100),
		fmt.Sprintf("Final miss rate   : %.2f%%", result.steadyMissRate*100),
	})
}

func printComparisonTable(results []runResult) {
	fmt.Fprintln(output, "")
	fmt.Fprintln(output, "=========================================================================================================")
	fmt.Fprintln(output, " THREE-WAY ALGORITHM PERFORMANCE BENCHMARK: CTD Baseline vs Original Paper vs Our Proposed Algorithm")
	fmt.Fprintln(output, "=========================================================================================================")
	fmt.Fprintln(output, " Algorithm          | Hit Rate | Byte Hit | Miss Rate | Evict Rate | Cache Util | Total Utility | Ops/sec")
	fmt.Fprintln(output, "--------------------+----------+----------+-----------+------------+------------+---------------+---------")

	var ctdResult, paperResult, oursResult *runResult
	for i := range results {
		r := &results[i]
		if r.name == "CTD" {
			ctdResult = r
		} else if r.name == "Paper-Proposed" {
			paperResult = r
		} else if r.name == "AdaLife-Pro" || r.name == "SMDP-DDQL" {
			oursResult = r
		}
	}

	ordered := []*runResult{ctdResult, paperResult, oursResult}
	for _, r := range ordered {
		if r == nil {
			continue
		}
		ops, _ := formatTiming(*r)
		s := r.stats
		displayName := r.name
		if r.name == "CTD" {
			displayName = "CTD (Baseline)"
		} else if r.name == "Paper-Proposed" {
			displayName = "Paper (Niknia '24)"
		} else if r.name == "AdaLife-Pro" {
			displayName = "AdaLife-Pro (Ours)"
		} else if r.name == "SMDP-DDQL" {
			displayName = "Our Part 1 SMDP"
		}
		fmt.Fprintf(output, " %-18s | %7.2f%% | %7.2f%% | %8.2f%% | %9.2f%% | %9.2f%% | %13.2f | %7s\n",
			displayName, s.HitRate()*100, s.ByteHitRate()*100, s.MissRate()*100, s.EvictionRate()*100, s.Utilization()*100, r.totalUtility, ops)
	}
	fmt.Fprintln(output, "=========================================================================================================")

	if oursResult != nil && paperResult != nil && ctdResult != nil {
		fmt.Fprintln(output, "")
		fmt.Fprintln(output, "  >>> KEY GAINS OF OUR PROPOSED ALGORITHM (AdaLife-Pro):")
		fmt.Fprintf(output, "      * Hit Rate Gain       : %+.2f%% vs Paper (Niknia '24) | %+.2f%% vs CTD Baseline\n",
			(oursResult.stats.HitRate()-paperResult.stats.HitRate())*100,
			(oursResult.stats.HitRate()-ctdResult.stats.HitRate())*100)
		fmt.Fprintf(output, "      * Byte Hit Rate Gain  : %+.2f%% vs Paper (Niknia '24) | %+.2f%% vs CTD Baseline\n",
			(oursResult.stats.ByteHitRate()-paperResult.stats.ByteHitRate())*100,
			(oursResult.stats.ByteHitRate()-ctdResult.stats.ByteHitRate())*100)
		fmt.Fprintf(output, "      * Eviction Reduction  : %+.2f%% vs Paper (Niknia '24) | %+.2f%% vs CTD Baseline\n",
			(oursResult.stats.EvictionRate()-paperResult.stats.EvictionRate())*100,
			(oursResult.stats.EvictionRate()-ctdResult.stats.EvictionRate())*100)
		fmt.Fprintf(output, "      * Total Utility Gain  : %+.2f points vs Paper          | %+.2f points vs CTD\n",
			oursResult.totalUtility-paperResult.totalUtility,
			oursResult.totalUtility-ctdResult.totalUtility)
		fmt.Fprintln(output, "=========================================================================================================")
	}
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
	fmt.Fprintln(w, "METRICS   : hit_rate | byte_hit_rate | miss_rate | eviction_rate | utilization | total_utility | avg_ns/request")
	for _, r := range results {
		_, latency := formatTiming(r)
		fmt.Fprintf(w, "%-10s : %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f | %s\n",
			r.name,
			r.stats.HitRate()*100,
			r.stats.ByteHitRate()*100,
			r.stats.MissRate()*100,
			r.stats.EvictionRate()*100,
			r.stats.Utilization()*100,
			r.totalUtility,
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
	printBox("GENERATED FILES OVERVIEW", []string{
		fmt.Sprintf("Count             : %d files", len(files)),
		fmt.Sprintf("Total size        : %.2f MiB", total),
		fmt.Sprintf("Average size      : %.2f MiB", total/float64(len(files))),
		fmt.Sprintf("Minimum size      : %.2f MiB", minimum),
		fmt.Sprintf("Maximum size      : %.2f MiB", maximum),
	})
}

func printBox(title string, lines []string) {
	width := 85
	for _, line := range lines {
		if len(line)+6 > width {
			width = len(line) + 6
		}
	}
	border := strings.Repeat("=", width)
	subBorder := strings.Repeat("-", width)
	fmt.Fprintln(output, "")
	fmt.Fprintln(output, border)
	fmt.Fprintf(output, "  %s\n", title)
	fmt.Fprintln(output, subBorder)
	for _, line := range lines {
		fmt.Fprintf(output, "    %s\n", line)
	}
	fmt.Fprintln(output, border)
}

func printTableIII(results []baselines.MDPEvaluationResult) {
	fmt.Fprintln(output, "")
	fmt.Fprintln(output, "=========================================================================================")
	fmt.Fprintln(output, " TABLE III: Hit Count for Discrete MDP vs Continuous SMDP (1000 Total Trials)")
	fmt.Fprintln(output, "=========================================================================================")
	fmt.Fprintln(output, " Time Quota (s)      | lambda = 5.00 req/s  | lambda = 1.66 req/s  | lambda = 1.00 req/s")
	fmt.Fprintln(output, "---------------------+----------------------+----------------------+---------------------")

	quotas := []float64{0.2, 0.4, 0.6, 0.8, 1.0}
	tableMap := make(map[float64]map[float64]int)
	smdpMap := make(map[float64]int)

	for _, r := range results {
		if _, ok := tableMap[r.Quota]; !ok {
			tableMap[r.Quota] = make(map[float64]int)
		}
		tableMap[r.Quota][r.Lambda] = r.MDPHitCount
		smdpMap[r.Lambda] = r.SMDPHitCount
	}

	for _, q := range quotas {
		fmt.Fprintf(output, " MDP  quota = %-6.2fs |      %-15d |      %-15d |      %-15d\n",
			q, tableMap[q][5.0], tableMap[q][1.66], tableMap[q][1.0])
	}
	fmt.Fprintln(output, "---------------------+----------------------+----------------------+---------------------")
	fmt.Fprintf(output, " SMDP (Continuous)   |      %-15d |      %-15d |      %-15d\n",
		smdpMap[5.0], smdpMap[1.66], smdpMap[1.0])
	fmt.Fprintln(output, "=========================================================================================")
}

