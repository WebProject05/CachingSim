package core

import (
	"math/rand"
	"smdp-edge-caching-framework/pkg/config"
)

// FileMetadata represents the static and dynamic characteristics of a file type f
// as formulated in Section III-A and Table II of the research paper.
type FileMetadata struct {
	ID         int     // Index f (0 to F-1)
	Size       float64 // z_f: File size in MiB, uniformly sampled in [100, 1000]
	Lifetime   float64 // w_l^f: Validity lifetime, uniformly sampled in [10, 30]
	Importance float64 // i_f: File importance coefficient, uniformly sampled in [0.1, 0.9]
	GenTime    float64 // w_g^f: Timestamp when the file was fetched/generated
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

// Freshness computes the age ratio h^f(t) = (currentTime - GenTime) / Lifetime in [0, 1].
func (f *FileMetadata) Freshness(currentTime float64) float64 {
	return CalculateFreshness(currentTime, f.GenTime, f.Lifetime)
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
