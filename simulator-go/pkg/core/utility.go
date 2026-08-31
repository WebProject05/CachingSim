package core

import (
	"math"
	"smdp-edge-caching-framework/pkg/config"
)

// CalculateFreshness computes the normalized age of a file at time t (Section III-A):
//
//	h^f(t) = (t - w_g^f) / w_l^f,  where 0 <= h^f(t) <= 1
//
// Parameters:
//   - currentTime: current simulation timestamp t
//   - genTime: timestamp when the file was fetched/generated w_g^f
//   - lifetime: total validity duration w_l^f
func CalculateFreshness(currentTime, genTime, lifetime float64) float64 {
	if lifetime <= 0 {
		return 1.0
	}
	if currentTime <= genTime {
		return 0.0
	}
	h := (currentTime - genTime) / lifetime
	if h < 0.0 {
		return 0.0
	}
	if h > 1.0 {
		return 1.0
	}
	return h
}

// CalculateUtility computes the non-linear utility y_f(t) for a file type f (Eq. 1):
//
//	y_f(t) = (-e^(h^f(t) + log(Curve)) + UT_max + Curve) * i_f
//	       = (-Curve * e^(h^f(t)) + UT_max + Curve) * i_f
//
// Where Curve = (UT_max - UT_min) / (e - 1).
// As freshness h ranges from 0 (brand new) to 1 (expired):
//   - At h = 0: y_f(t) = UT_max * i_f
//   - At h = 1: y_f(t) = UT_min * i_f
func CalculateUtility(freshness, importance float64, cfg *config.Config) float64 {
	if importance <= 0 {
		return 0.0
	}
	h := freshness
	if h < 0.0 {
		h = 0.0
	} else if h > 1.0 {
		h = 1.0
	}

	curve := cfg.Curve
	if curve <= 0 {
		curve = config.ComputeCurve(cfg.UTMax, cfg.UTMin)
	}

	val := -curve*math.Exp(h) + cfg.UTMax + curve
	if val < cfg.UTMin {
		val = cfg.UTMin
	}
	return val * importance
}

// CalculateFileUtility calculates the utility of a FileMetadata at the given simulation time.
func CalculateFileUtility(file *FileMetadata, currentTime float64, cfg *config.Config) float64 {
	freshness := CalculateFreshness(currentTime, file.GenTime, file.Lifetime)
	return CalculateUtility(freshness, file.Importance, cfg)
}

// CalculateAllUtilities computes utility y_f(t) for all F file types at time t.
func CalculateAllUtilities(files []FileMetadata, currentTime float64, cfg *config.Config) []float64 {
	utilities := make([]float64, len(files))
	for i := range files {
		utilities[i] = CalculateFileUtility(&files[i], currentTime, cfg)
	}
	return utilities
}

// CalculateCachedWorth computes total worth W(t) of cached files (Eq. 3):
//
//	W(t) = (b(t) . d(t)) (b(t) . y(t))^T = sum_{f=1}^F b_f(t) * d_f(t) * y_f(t)
func CalculateCachedWorth(cached []bool, popularity []float64, utilities []float64) float64 {
	var worth float64
	limit := len(cached)
	if len(popularity) < limit {
		limit = len(popularity)
	}
	if len(utilities) < limit {
		limit = len(utilities)
	}
	for i := 0; i < limit; i++ {
		if cached[i] {
			worth += popularity[i] * utilities[i]
		}
	}
	return worth
}
