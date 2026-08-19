package tests

import (
	"math/rand"
	"testing"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

func TestEvictUntilFits(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.CacheCapacity = 1000.0 // Set small capacity for test

	rng := rand.New(rand.NewSource(42))
	files := core.GenerateFiles(cfg, rng)

	// Fix sizes
	files[0].Size = 400.0
	files[1].Size = 400.0
	files[2].Size = 400.0

	// Set importance so file 0 has the lowest utility
	files[0].Importance = 0.1
	files[1].Importance = 0.8
	files[2].Importance = 0.9

	cache := core.NewCacheEngine(cfg, files)
	cache.Cached[0] = true
	cache.Cached[1] = true
	cache.UsedCapacity = 800.0

	// Inserting file 2 (size 400) requires evicting file 0
	cache.EvictUntilFits(2)

	if cache.Cached[0] {
		t.Errorf("Expected file 0 (lowest utility) to be evicted, but it is still cached")
	}
	if !cache.Cached[1] {
		t.Errorf("Expected file 1 to remain in cache")
	}
	if (cache.Cfg.CacheCapacity - cache.UsedCapacity) < files[2].Size {
		t.Errorf("Available capacity %.2f is insufficient for file size %.2f",
			cache.Cfg.CacheCapacity-cache.UsedCapacity, files[2].Size)
	}
}

func TestRewardCalculation(t *testing.T) {
	cfg := config.DefaultConfig()
	rng := rand.New(rand.NewSource(42))
	files := core.GenerateFiles(cfg, rng)

	cache := core.NewCacheEngine(cfg, files)
	cache.RecordRequest(0)
	cache.Cached[0] = true
	cache.UsedCapacity = files[0].Size

	reward := cache.ComputeReward()
	if reward == 0.0 {
		t.Errorf("Reward should not be zero after recording requests and caching item")
	}
}