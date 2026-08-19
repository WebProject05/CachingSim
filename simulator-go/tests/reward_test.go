package tests

import (
	"math"
	"testing"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

func TestRewardIsFiniteWithZeroCapacity(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.CacheCapacity = 0
	cache := core.NewCacheEngine(cfg, []core.FileMetadata{{ID: 0, Size: 1}})

	reward := cache.ComputeReward()
	if math.IsInf(reward, 0) || math.IsNaN(reward) {
		t.Fatalf("reward must remain finite, got %v", reward)
	}
}
