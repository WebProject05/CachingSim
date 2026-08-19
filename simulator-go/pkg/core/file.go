package core

import (
	"math"
	"math/rand"
	"smdp-edge-caching-framework/pkg/config"
)

type FileMetadata struct {
	ID         int     // Index f (0 to F-1)
	Size       float64 // z_f in [100, 1000]
	Lifetime   float64 // w_l^f in [10, 30]
	Importance float64 // i_f in [0.1, 0.9]
	GenTime    float64 // w_g^f (timestamp when generated)
}

// GenerateFiles initializes F file types with random attributes as per Table II
func GenerateFiles(cfg *config.Config, r *rand.Rand) []FileMetadata {
	files := make([]FileMetadata, cfg.TotalFileTypes)
	for i := 0; i < cfg.TotalFileTypes; i++ {
		files[i] = FileMetadata{
			ID:         i,
			Size:       100.0 + r.Float64()*(1000.0-100.0), // Uniform [100, 1000]
			Lifetime:   10.0 + r.Float64()*(30.0-10.0),     // Uniform [10, 30]
			Importance: 0.1 + r.Float64()*(0.9-0.1),        // Uniform [0.1, 0.9]
			GenTime:    0.0,
		}
	}
	return files
}

// Freshness computes h^f(t) = (t - w_g^f) / w_l^f, bounded in [0, 1]
func (f *FileMetadata) Freshness(currentTime float64) float64 {
	if f.Lifetime <= 0 {
		return 1.0
	}
	h := (currentTime - f.GenTime) / f.Lifetime
	if h < 0.0 {
		return 0.0
	}
	if h > 1.0 {
		return 1.0
	}
	return h
}

// Utility computes y_f(t) based on Eq. (1)
func (f *FileMetadata) Utility(currentTime float64, cfg *config.Config) float64 {
	h := f.Freshness(currentTime)
	// If file has completely expired (h >= 1), utility drops to minimum
	val := -math.Exp(h+math.Log(cfg.Curve)) + cfg.UTMax + cfg.Curve
	return val * f.Importance
}