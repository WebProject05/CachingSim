package core

import (
	"math"
	"smdp-edge-caching-framework/pkg/config"
)

// Evictor defines the interface for cache eviction strategies.
type Evictor interface {
	// Evict selects and removes files until sufficient space is freed for requiredSize.
	// Returns the slice of evicted file IDs.
	Evict(cached []bool, files []FileMetadata, usedCapacity, cacheCapacity, requiredSize, currentTime float64, cfg *config.Config) ([]int, float64)
}

// LowestUtilityEvictor implements the paper's eviction policy (Section IV-A-2):
// It repeatedly removes the file currently possessing the lowest utility y_f(t)
// until enough capacity is freed to accommodate the newly requested file.
type LowestUtilityEvictor struct{}

// NewLowestUtilityEvictor constructs a new LowestUtilityEvictor.
func NewLowestUtilityEvictor() *LowestUtilityEvictor {
	return &LowestUtilityEvictor{}
}

// Evict iteratively finds and removes the cached file with minimum utility
// until (cacheCapacity - usedCapacity) >= requiredSize.
func (e *LowestUtilityEvictor) Evict(
	cached []bool,
	files []FileMetadata,
	usedCapacity, cacheCapacity, requiredSize, currentTime float64,
	cfg *config.Config,
) ([]int, float64) {
	if cacheCapacity <= 0 || requiredSize > cacheCapacity {
		return nil, usedCapacity
	}

	var evictedIDs []int
	currentUsed := usedCapacity

	for (cacheCapacity - currentUsed) < requiredSize {
		lowestID := -1
		minUtility := math.MaxFloat64

		// Find cached item with minimum utility y_f(t)
		for id := 0; id < len(cached) && id < len(files); id++ {
			if !cached[id] {
				continue
			}
			u := files[id].Utility(currentTime, cfg)
			if u < minUtility {
				minUtility = u
				lowestID = id
			}
		}

		// If no items remain to evict, break
		if lowestID == -1 {
			break
		}

		cached[lowestID] = false
		currentUsed -= files[lowestID].Size
		if currentUsed < 0 {
			currentUsed = 0
		}
		evictedIDs = append(evictedIDs, lowestID)
	}

	return evictedIDs, currentUsed
}
