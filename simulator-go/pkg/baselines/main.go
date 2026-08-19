package baselines

import (
	"fmt"
	"math/rand"
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
	"smdp-edge-caching-framework/pkg/smdp"
	"time"
)

func main() {
	cfg := config.DefaultConfig()
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	// 1. Initialize System Components
	files := core.GenerateFiles(cfg, rng)
	cache := core.NewCacheEngine(cfg, files)
	generator := smdp.NewGenerator(cfg, seed)

	totalTrials := 1000
	hitCount := 0
	totalUtility := 0.0
	totalReward := 0.0

	fmt.Println("=== Starting SMDP Edge Caching Simulation (Go Native Engine) ===")
	fmt.Printf("File Types: %d, Capacity: %.0f, Lambda: %.2f\n\n", cfg.TotalFileTypes, cfg.CacheCapacity, cfg.LambdaSource)

	for trial := 1; trial <= totalTrials; trial++ {
		// 1. Generate Poisson arrival interval tau and advance time
		tau := generator.NextPoissonInterval(cfg.LambdaSource)
		cache.CurrentTime += tau

		// 2. Sample incoming request f_r via Zipf distribution
		reqFile := generator.SampleRequestedFile()
		cache.RecordRequest(reqFile)

		// 3. Check Cache Hit
		isHit := cache.Cached[reqFile]
		if isHit {
			hitCount++
			totalUtility += files[reqFile].Utility(cache.CurrentTime, cfg)
		}

		// 4. Decision: For now, benchmark using heuristic a(t)=1 (Cache all)
		action := 1
		if action == 1 && !isHit {
			cache.EvictUntilFits(reqFile)
			cache.Cached[reqFile] = true
			cache.UsedCapacity += files[reqFile].Size
		}

		// 5. Compute Reward r(t)
		r := cache.ComputeReward()
		totalReward += r

		if trial%200 == 0 {
			state := cache.GetCurrentState(reqFile)
			fmt.Printf("Trial [%4d/%4d] | Hits: %3d | Unused Mem: %5.1f%% | Instant Reward: %6.2f | Avg Reward: %6.2f\n",
				trial, totalTrials, hitCount, state.Mem*100.0, r, totalReward/float64(trial))
		}
	}

	fmt.Println("\n=== Final Simulation Metrics ===")
	fmt.Printf("Total Hits: %d / %d (%.2f%%)\n", hitCount, totalTrials, float64(hitCount)/float64(totalTrials)*100.0)
	fmt.Printf("Accumulated Total Utility: %.2f\n", totalUtility)
	fmt.Printf("Average Reward in Long Run: %.2f\n", totalReward/float64(totalTrials))
}
