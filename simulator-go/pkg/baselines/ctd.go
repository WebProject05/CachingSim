package baselines

import (
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

// CTD tracks freshness and availability of current request only (Zhu et al. [17])
type CTDBaseline struct {
	cfg          *config.Config
	files        []core.FileMetadata
	cached       map[int]bool
	usedCapacity float64
}

func NewCTDBaseline(cfg *config.Config, files []core.FileMetadata) *CTDBaseline {
	return &CTDBaseline{
		cfg:          cfg,
		files:        files,
		cached:       make(map[int]bool),
		usedCapacity: 0.0,
	}
}

// ComputeCTDReward calculates reward focusing only on the current requested item
func (c *CTDBaseline) ComputeCTDReward(fileID int, currentTime float64, isHit bool) float64 {
	if !isHit {
		return -1.0 // Transmission delay penalty from data center
	}
	freshness := c.files[fileID].Freshness(currentTime)
	return 1.0 - freshness // Instant freshness reward
}