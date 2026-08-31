package baselines

import (
	"testing"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

func testFiles(sizes ...float64) []core.FileMetadata {
	files := make([]core.FileMetadata, len(sizes))
	for id, size := range sizes {
		files[id] = core.FileMetadata{
			ID:         id,
			Size:       size,
			Lifetime:   20.0,
			Importance: 0.5,
			GenTime:    0.0,
		}
	}
	return files
}

func testConfig(capacity float64) *config.Config {
	cfg := config.DefaultConfig()
	cfg.CacheCapacity = capacity
	return cfg
}

func TestLRUAccessAndEviction(t *testing.T) {
	cache := NewLRUCache(testConfig(10), testFiles(4, 4, 4))

	if cache.Access(0) || cache.Access(1) {
		t.Fatal("first accesses should be misses")
	}
	if !cache.Access(0) {
		t.Fatal("re-accessing a cached file should be a hit")
	}
	cache.Access(2)
	if cache.inCache[1] {
		t.Fatal("least recently used file was not evicted")
	}
	if !cache.inCache[0] {
		t.Fatal("most recently used file was evicted")
	}
}

func TestFIFOEvictsOldestInsertion(t *testing.T) {
	cache := NewFIFOCache(testConfig(8), testFiles(4, 4, 4))

	cache.Access(0)
	cache.Access(1)
	cache.Access(0)
	cache.Access(2)

	if _, ok := cache.entries[0]; ok {
		t.Fatal("FIFO evicted a newer insertion instead of the oldest file")
	}
	if _, ok := cache.entries[1]; !ok {
		t.Fatal("FIFO did not retain the newer insertion")
	}
	if cache.Stats().Evictions != 1 || cache.Stats().Hits != 1 {
		t.Fatalf("unexpected FIFO stats: %+v", cache.Stats())
	}
}

func TestLFUFrequencyAndDeterministicTieBreak(t *testing.T) {
	cache := NewLFUCache(testConfig(8), testFiles(4, 4, 4))

	cache.Access(0)
	cache.Access(1)
	cache.Access(0)
	cache.Access(2)

	if !cache.cached[0] || !cache.cached[2] || cache.cached[1] {
		t.Fatal("LFU did not evict the least frequent, oldest item")
	}
	if cache.frequencies[1] != 0 {
		t.Fatal("evicted frequency metadata was not removed")
	}
}

func TestBaselineRejectsInvalidAndOversizedFiles(t *testing.T) {
	files := testFiles(4, 12)
	for _, cache := range []interface{ Access(int) bool }{
		NewLRUCache(testConfig(10), files),
		NewLFUCache(testConfig(10), files),
		NewSIEVECache(testConfig(10), files),
		NewCTDCache(testConfig(10), files),
	} {
		if cache.Access(-1) || cache.Access(len(files)) || cache.Access(1) {
			t.Fatal("invalid or oversized access should be a miss")
		}
	}
}

func TestBaselineStats(t *testing.T) {
	cache := NewLRUCache(testConfig(8), testFiles(4, 4, 4))
	cache.Access(0)
	cache.Access(0)
	cache.Access(1)
	cache.Access(2)

	stats := cache.Stats()
	if stats.Requests != 4 || stats.Hits != 1 || stats.Misses != 3 || stats.Evictions != 1 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	if stats.HitRate() != 0.25 || stats.Utilization() != 1 {
		t.Fatalf("unexpected derived stats: hit rate %.2f, utilization %.2f", stats.HitRate(), stats.Utilization())
	}
	if stats.Insertions != 3 || stats.RequestedBytes != 16 || stats.HitBytes != 4 || stats.MissBytes != 12 {
		t.Fatalf("unexpected byte stats: %+v", stats)
	}
	if stats.MissRate() != 0.75 || stats.ByteHitRate() != 0.25 || stats.AverageRequestBytes() != 4 {
		t.Fatalf("unexpected comparison metrics: miss rate %.2f, byte hit rate %.2f, average request %.2f", stats.MissRate(), stats.ByteHitRate(), stats.AverageRequestBytes())
	}
}

func TestSIEVEUsesReferenceBitAndEvicts(t *testing.T) {
	cache := NewSIEVECache(testConfig(8), testFiles(4, 4, 4))
	cache.Access(0)
	cache.Access(1)
	cache.Access(0)
	cache.Access(2)

	if cache.cachedCount != 2 || !cache.inCache[0] || !cache.inCache[2] {
		t.Fatalf("SIEVE evicted a referenced item: %+v", cache.Stats())
	}
	if cache.inCache[1] || cache.Stats().Evictions != 1 {
		t.Fatalf("SIEVE did not evict the unreferenced item: %+v", cache.Stats())
	}
}

func TestCTDCacheHitAndEviction(t *testing.T) {
	files := testFiles(4, 4, 4)
	cache := NewCTDCache(testConfig(8), files)

	// Access file 0 at t=0 (miss)
	if cache.AccessAtTime(0, 0.0) {
		t.Fatal("first access should be a miss")
	}
	// Access file 0 at t=2 (hit, fresh)
	if !cache.AccessAtTime(0, 2.0) {
		t.Fatal("access before lifetime expiration should be a hit")
	}

	// Access file 1 at t=3 (miss, caches file 1)
	cache.AccessAtTime(1, 3.0)

	// Access file 2 at t=5 (miss, cache full -> evicts least fresh item)
	cache.AccessAtTime(2, 5.0)

	stats := cache.Stats()
	if stats.Hits != 1 || stats.Requests != 4 {
		t.Fatalf("unexpected CTD stats: %+v", stats)
	}
	if cache.TotalReward() == 0.0 {
		t.Fatal("total CTD reward should be non-zero")
	}
}

func TestRunSMDPSimulation(t *testing.T) {
	cfg := testConfig(1000)
	cfg.TotalFileTypes = 5
	result := RunSMDPSimulation(cfg, 42, 100)

	if result.TotalTrials != 100 {
		t.Errorf("expected 100 total trials, got %d", result.TotalTrials)
	}
	if result.HitCount < 0 || result.HitCount > 100 {
		t.Errorf("invalid hit count: %d", result.HitCount)
	}
}

func TestRunMDPvsSMDPComparison(t *testing.T) {
	cfg := testConfig(5000)
	cfg.TotalFileTypes = 10
	files := core.GenerateFiles(cfg, nil)
	// Fallback if nil rng
	for i := range files {
		files[i] = core.FileMetadata{ID: i, Size: 500, Lifetime: 20, Importance: 0.5}
	}

	results := RunMDPvsSMDPComparison(cfg, files, 42, 100, []float64{1.0}, []float64{0.2, 0.5})
	if len(results) != 2 {
		t.Fatalf("expected 2 comparison rows, got %d", len(results))
	}
	for _, r := range results {
		if r.MDPHitCount < 0 || r.SMDPHitCount < 0 {
			t.Errorf("invalid hit counts: %+v", r)
		}
	}
}
