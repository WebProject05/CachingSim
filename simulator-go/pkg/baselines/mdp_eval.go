package baselines

import (
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
	"smdp-edge-caching-framework/pkg/smdp"
)

// MDPEvaluationResult holds hit count and utility for discrete MDP vs continuous SMDP.
type MDPEvaluationResult struct {
	Quota        float64 // Fixed time quota (seconds) for discrete MDP
	Lambda       float64 // Poisson request rate
	MDPHitCount  int     // Number of cache hits out of totalTrials for MDP
	SMDPHitCount int     // Number of cache hits out of totalTrials for SMDP
	MDPHitRate   float64 // MDP hit rate %
	SMDPHitRate  float64 // SMDP hit rate %
}

// RunMDPvsSMDPComparison reproduces the experimental comparison in Section VII-E-1-i and Table III:
// Compares hit rates for discrete-time MDP under varying time quotas vs continuous-time SMDP.
func RunMDPvsSMDPComparison(
	cfg *config.Config,
	files []core.FileMetadata,
	seed int64,
	totalTrials int,
	lambdas []float64,
	quotas []float64,
) []MDPEvaluationResult {
	if totalTrials <= 0 {
		totalTrials = 1000
	}
	if len(lambdas) == 0 {
		lambdas = []float64{5.0, 1.66, 1.0, 0.2}
	}
	if len(quotas) == 0 {
		quotas = []float64{0.2, 0.4, 0.6, 0.8, 1.0}
	}

	var results []MDPEvaluationResult

	for _, lambda := range lambdas {
		// Run SMDP baseline for this lambda
		smdpCfg := *cfg
		smdpCfg.LambdaSource = lambda
		smdpFiles := core.CloneFiles(files)
		smdpEngine := core.NewCacheEngine(&smdpCfg, smdpFiles)
		smdpGen := smdp.NewGenerator(&smdpCfg, seed)

		smdpHits := 0
		for t := 0; t < totalTrials; t++ {
			tau := smdpGen.NextPoissonInterval(lambda)
			smdpEngine.CurrentTime += tau

			reqFile := smdpGen.SampleRequestedFile()
			smdpEngine.RecordRequest(reqFile)

			if smdpEngine.IsCached(reqFile) {
				smdpHits++
			} else {
				smdpEngine.EvictUntilFits(reqFile)
				smdpEngine.Insert(reqFile)
			}
		}

		// Run MDP with each time quota
		for _, quota := range quotas {
			mdpCfg := *cfg
			mdpCfg.LambdaSource = lambda
			mdpFiles := core.CloneFiles(files)
			mdpEngine := core.NewCacheEngine(&mdpCfg, mdpFiles)
			mdpGen := smdp.NewGenerator(&mdpCfg, seed)

			mdpHits := 0
			currentTime := 0.0

			for t := 0; t < totalTrials; t++ {
				// In discrete MDP, decisions happen at synchronized time quotas
				tau := mdpGen.NextPoissonInterval(lambda)
				currentTime += tau

				// Align simulation time with next time quota boundary
				quotaSteps := int(currentTime / quota)
				mdpEngine.CurrentTime = float64(quotaSteps+1) * quota

				reqFile := mdpGen.SampleRequestedFile()
				mdpEngine.RecordRequest(reqFile)

				if mdpEngine.IsCached(reqFile) {
					mdpHits++
				} else {
					mdpEngine.EvictUntilFits(reqFile)
					mdpEngine.Insert(reqFile)
				}
			}

			results = append(results, MDPEvaluationResult{
				Quota:        quota,
				Lambda:       lambda,
				MDPHitCount:  mdpHits,
				SMDPHitCount: smdpHits,
				MDPHitRate:   float64(mdpHits) / float64(totalTrials) * 100.0,
				SMDPHitRate:  float64(smdpHits) / float64(totalTrials) * 100.0,
			})
		}
	}

	return results
}
