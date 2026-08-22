package baselines

import (
	"testing"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

func testFiles(sizes ...float64) []core.FileMetadata {
	files := make([]core.FileMetadata, len(sizes))
	for id, size := range sizes {
		files[id] = core.FileMetadata{ID: id, Size: size}
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
}

func TestMissesDecreaseForRepeatedLocalWorkload(t *testing.T) {
	for _, newCache := range []func() interface{ Access(int) bool }{
		func() interface{ Access(int) bool } { return NewLRUCache(testConfig(8), testFiles(4, 4)) },
		func() interface{ Access(int) bool } { return NewLFUCache(testConfig(8), testFiles(4, 4)) },
		func() interface{ Access(int) bool } { return NewSIEVECache(testConfig(8), testFiles(4, 4)) },
	} {
		cache := newCache()
		warmupMisses := 0
		steadyMisses := 0
		for request := 0; request < 20; request++ {
			fileID := request % 2
			if !cache.Access(fileID) {
				if request < 4 {
					warmupMisses++
				}
				if request >= 16 {
					steadyMisses++
				}
			}
		}
		if steadyMisses >= warmupMisses {
			t.Fatalf("expected repeated workload misses to decrease: warmup=%d steady=%d", warmupMisses, steadyMisses)
		}
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

func TestLFUHeapStaysBoundedByCachedFiles(t *testing.T) {
	cache := NewLFUCache(testConfig(8), testFiles(4, 4))
	for request := 0; request < 1000; request++ {
		cache.Access(request % 2)
	}

	if cache.evictionHeap.Len() != cache.cachedCount {
		t.Fatalf("LFU heap grew beyond cached files: heap=%d cached=%d", cache.evictionHeap.Len(), cache.cachedCount)
	}
}
