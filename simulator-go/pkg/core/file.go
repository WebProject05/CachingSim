package core

import (
	"math"
	"math/rand"
	"smdp-edge-caching-framework/pkg/config"
)

// FileMetadata represents the static and dynamic characteristics of a file type f
// as formulated in Section III-A and Table II of the research paper, extended with
// AdaLife-Pro dynamic lease tracking and fetch-cost attributes.
type FileMetadata struct {
	ID                int     // Index f (0 to F-1)
	Size              float64 // z_f: File size in MiB, uniformly sampled in [100, 1000]
	Lifetime          float64 // w_l^f: Base validity lifetime, uniformly sampled in [10, 30]
	EffectiveLifetime float64 // Dynamic TTL adapted by AdaLife-Pro lease extensions
	Importance        float64 // i_f: File importance coefficient, uniformly sampled in [0.1, 0.9]
	GenTime           float64 // w_g^f: Timestamp when the file was fetched or renewed
	HitCount          int     // Number of cache hits during the current resident residency
}

// GenerateFiles initializes F file types with random attributes as per Table II:
//   - Size: [100, 1000] MiB
//   - Lifetime: [10, 30]
//   - Importance: [0.1, 0.9]
//   - GenTime: 0.0
func GenerateFiles(cfg *config.Config, r *rand.Rand) []FileMetadata {
	if cfg.TotalFileTypes <= 0 {
		return nil
	}
	if r == nil {
		r = rand.New(rand.NewSource(42))
	}
	files := make([]FileMetadata, cfg.TotalFileTypes)
	for i := 0; i < cfg.TotalFileTypes; i++ {
		files[i] = FileMetadata{
			ID:         i,
			Size:       cfg.FileSizeMin + r.Float64()*(cfg.FileSizeMax-cfg.FileSizeMin),
			Lifetime:   cfg.LifetimeMin + r.Float64()*(cfg.LifetimeMax-cfg.LifetimeMin),
			Importance: cfg.ImportanceMin + r.Float64()*(cfg.ImportanceMax-cfg.ImportanceMin),
			GenTime:    0.0,
		}
	}
	return files
}

// GenerateSweepFiles creates file metadata for parameter sweep experiments (Section VII-E, Figs. 4 & 5):
// All other files (f > 0) have default fixed properties (lifetime=20, size=500, importance=0.5),
// while the most popular file (f=0) receives custom values.
func GenerateSweepFiles(cfg *config.Config, popLifetime, popSize, popImportance float64) []FileMetadata {
	files := make([]FileMetadata, cfg.TotalFileTypes)
	for i := 0; i < cfg.TotalFileTypes; i++ {
		if i == 0 {
			files[i] = FileMetadata{
				ID:         0,
				Size:       popSize,
				Lifetime:   popLifetime,
				Importance: popImportance,
				GenTime:    0.0,
			}
		} else {
			files[i] = FileMetadata{
				ID:         i,
				Size:       500.0,
				Lifetime:   20.0,
				Importance: 0.5,
				GenTime:    0.0,
			}
		}
	}
	return files
}

// ActiveLifetime returns the effective dynamic TTL if active, falling back to base Lifetime.
func (f *FileMetadata) ActiveLifetime() float64 {
	if f.EffectiveLifetime > 0 {
		return f.EffectiveLifetime
	}
	return f.Lifetime
}

// Freshness computes the age ratio h^f(t) = (currentTime - GenTime) / ActiveLifetime in [0, 1].
func (f *FileMetadata) Freshness(currentTime float64) float64 {
	return CalculateFreshness(currentTime, f.GenTime, f.ActiveLifetime())
}

// FetchCost computes the origin retrieval penalty C_fetch(f) = RTT + bw * (z_f / 1000).
// Resolving Gap #11 (Fetch-Cost & Latency Asymmetry).
func (f *FileMetadata) FetchCost(cfg *config.Config) float64 {
	if cfg == nil || !cfg.EnableFetchCost {
		return 1.0
	}
	return cfg.BaseOriginRTT + cfg.FetchBandwidthWeight*(f.Size/1000.0)
}

// ResetLease initializes a fresh TTL lease when fetched into cache.
func (f *FileMetadata) ResetLease(currentTime, scale float64) {
	if scale <= 0 {
		scale = 1.0
	}
	f.GenTime = currentTime
	f.EffectiveLifetime = f.Lifetime * scale
	f.HitCount = 1
}

// RenewLease extends the file's validity lifetime on a cache hit based on importance and hit frequency
// (AdaLife-Pro Layer 1: Dynamic Lifetime).
func (f *FileMetadata) RenewLease(currentTime float64, cfg *config.Config) {
	if cfg == nil || !cfg.EnableAdaLife {
		return
	}
	f.HitCount++
	cappedHits := math.Min(5.0, float64(f.HitCount))
	boost := 1.0 + cfg.LeaseImportanceWeight*f.Importance + cfg.LeaseHitWeight*cappedHits
	newTTL := f.Lifetime * boost
	if cfg.MaxLifetimeCap > 0 && newTTL > cfg.MaxLifetimeCap {
		newTTL = cfg.MaxLifetimeCap
	}
	f.EffectiveLifetime = newTTL
	f.GenTime = currentTime
}

// Utility computes non-linear utility y_f(t) based on freshness and importance (Eq. 1).
func (f *FileMetadata) Utility(currentTime float64, cfg *config.Config) float64 {
	return CalculateFileUtility(f, currentTime, cfg)
}

// CloneFiles deep copies a slice of FileMetadata.
func CloneFiles(files []FileMetadata) []FileMetadata {
	if files == nil {
		return nil
	}
	copied := make([]FileMetadata, len(files))
	copy(copied, files)
	return copied
}
