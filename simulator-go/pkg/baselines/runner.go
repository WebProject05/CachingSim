package baselines

import (
	"math/rand"
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
	"smdp-edge-caching-framework/pkg/smdp"
)

// SimulationResult holds the summary metrics from a simulation run.
type SimulationResult struct {
	TotalTrials  int
	HitCount     int
	HitRate      float64
	TotalUtility float64
	TotalReward  float64
	AvgReward    float64
}

// RunSMDPSimulation executes a reactive SMDP caching simulation for a given configuration.
func RunSMDPSimulation(cfg *config.Config, seed int64, totalTrials int) SimulationResult {
	if totalTrials <= 0 {
		totalTrials = 1000
	}
	rng := rand.New(rand.NewSource(seed))
	files := core.GenerateFiles(cfg, rng)
	cache := core.NewCacheEngine(cfg, files)
	generator := smdp.NewGenerator(cfg, seed)

	hitCount := 0
	totalUtility := 0.0
	totalReward := 0.0

	for trial := 1; trial <= totalTrials; trial++ {
		tau := generator.NextPoissonInterval(cfg.LambdaSource)
		cache.CurrentTime += tau

		reqFile := generator.SampleRequestedFile()
		cache.RecordRequest(reqFile)

		isHit := cache.IsCached(reqFile)
		if isHit {
			hitCount++
			if reqFile >= 0 && reqFile < len(files) {
				totalUtility += files[reqFile].Utility(cache.CurrentTime, cfg)
			}
		} else {
			cache.EvictUntilFits(reqFile)
			cache.Insert(reqFile)
		}

		r := cache.ComputeReward()
		totalReward += r
	}

	return SimulationResult{
		TotalTrials:  totalTrials,
		HitCount:     hitCount,
		HitRate:      float64(hitCount) / float64(totalTrials) * 100.0,
		TotalUtility: totalUtility,
		TotalReward:  totalReward,
		AvgReward:    totalReward / float64(totalTrials),
	}
}
