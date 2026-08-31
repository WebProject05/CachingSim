package core

import (
	"math"
	"math/rand"
	"testing"

	"smdp-edge-caching-framework/pkg/config"
)

func TestFileGeneration(t *testing.T) {
	cfg := config.DefaultConfig()
	rng := rand.New(rand.NewSource(42))
	files := GenerateFiles(cfg, rng)

	if len(files) != cfg.TotalFileTypes {
		t.Fatalf("expected %d files, got %d", cfg.TotalFileTypes, len(files))
	}

	for i, f := range files {
		if f.ID != i {
			t.Errorf("file %d has incorrect ID %d", i, f.ID)
		}
		if f.Size < cfg.FileSizeMin || f.Size > cfg.FileSizeMax {
			t.Errorf("file %d size %f out of bounds [%f, %f]", i, f.Size, cfg.FileSizeMin, cfg.FileSizeMax)
		}
		if f.Lifetime < cfg.LifetimeMin || f.Lifetime > cfg.LifetimeMax {
			t.Errorf("file %d lifetime %f out of bounds [%f, %f]", i, f.Lifetime, cfg.LifetimeMin, cfg.LifetimeMax)
		}
		if f.Importance < cfg.ImportanceMin || f.Importance > cfg.ImportanceMax {
			t.Errorf("file %d importance %f out of bounds [%f, %f]", i, f.Importance, cfg.ImportanceMin, cfg.ImportanceMax)
		}
	}
}

func TestSweepFilesGeneration(t *testing.T) {
	cfg := config.DefaultConfig()
	files := GenerateSweepFiles(cfg, 25.0, 750.0, 0.8)

	if len(files) != cfg.TotalFileTypes {
		t.Fatalf("expected %d files, got %d", cfg.TotalFileTypes, len(files))
	}
	if files[0].Lifetime != 25.0 || files[0].Size != 750.0 || files[0].Importance != 0.8 {
		t.Errorf("popular file (index 0) properties incorrect: %+v", files[0])
	}
	if files[1].Lifetime != 20.0 || files[1].Size != 500.0 || files[1].Importance != 0.5 {
		t.Errorf("non-popular file properties incorrect: %+v", files[1])
	}
}

func TestFreshnessAndUtility(t *testing.T) {
	cfg := config.DefaultConfig()
	file := &FileMetadata{
		ID:         0,
		Size:       500.0,
		Lifetime:   20.0,
		Importance: 0.8,
		GenTime:    10.0,
	}

	// Case 1: Brand new file (currentTime = GenTime = 10.0) -> freshness = 0
	h0 := file.Freshness(10.0)
	if h0 != 0.0 {
		t.Errorf("expected freshness 0.0 at genTime, got %f", h0)
	}
	u0 := file.Utility(10.0, cfg)
	expectedU0 := cfg.UTMax * file.Importance // 1.5 * 0.8 = 1.2
	if math.Abs(u0-expectedU0) > 1e-6 {
		t.Errorf("expected max utility %f at h=0, got %f", expectedU0, u0)
	}

	// Case 2: Fully expired file (currentTime = 30.0) -> freshness = 1.0
	h1 := file.Freshness(30.0)
	if h1 != 1.0 {
		t.Errorf("expected freshness 1.0 at expiration, got %f", h1)
	}
	u1 := file.Utility(30.0, cfg)
	expectedU1 := cfg.UTMin * file.Importance // 0.1 * 0.8 = 0.08
	if math.Abs(u1-expectedU1) > 1e-6 {
		t.Errorf("expected min utility %f at h=1, got %f", expectedU1, u1)
	}

	// Case 3: Over-expired file (currentTime = 50.0) -> clamped to 1.0
	hOver := file.Freshness(50.0)
	if hOver != 1.0 {
		t.Errorf("expected clamped freshness 1.0, got %f", hOver)
	}

	// Case 4: Monotonic decrease test
	uPrev := u0
	for ct := 11.0; ct <= 30.0; ct += 1.0 {
		uCurr := file.Utility(ct, cfg)
		if uCurr >= uPrev {
			t.Errorf("utility did not strictly decrease: at t=%f got %f >= %f", ct, uCurr, uPrev)
		}
		uPrev = uCurr
	}
}

func TestLowestUtilityEviction(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.CacheCapacity = 1000.0

	files := []FileMetadata{
		{ID: 0, Size: 400.0, Lifetime: 20.0, Importance: 0.1, GenTime: 0.0}, // Lowest utility
		{ID: 1, Size: 400.0, Lifetime: 20.0, Importance: 0.5, GenTime: 0.0}, // Medium utility
		{ID: 2, Size: 400.0, Lifetime: 20.0, Importance: 0.9, GenTime: 0.0}, // High utility (incoming)
	}

	cache := NewCacheEngine(cfg, files)
	cache.Insert(0)
	cache.Insert(1)

	if !cache.IsCached(0) || !cache.IsCached(1) {
		t.Fatal("files 0 and 1 should be cached")
	}

	// Insert file 2 which requires 400 MiB (current used = 800, remaining = 200)
	evicted := cache.EvictUntilFits(2)
	if len(evicted) != 1 || evicted[0] != 0 {
		t.Fatalf("expected file 0 to be evicted, got: %v", evicted)
	}
	if cache.IsCached(0) {
		t.Error("file 0 should no longer be cached")
	}
	if !cache.IsCached(1) {
		t.Error("file 1 should remain cached")
	}

	inserted := cache.Insert(2)
	if !inserted || !cache.IsCached(2) {
		t.Error("file 2 should be successfully inserted")
	}
}

func TestSlidingWindowPopularity(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.SlidingWindowN = 3
	cfg.TotalFileTypes = 4

	files := []FileMetadata{
		{ID: 0, Size: 100},
		{ID: 1, Size: 100},
		{ID: 2, Size: 100},
		{ID: 3, Size: 100},
	}

	cache := NewCacheEngine(cfg, files)

	// Record requests: 0, 1, 0
	cache.RecordRequest(0)
	cache.RecordRequest(1)
	cache.RecordRequest(0)

	pop := cache.GetPopularityVector()
	if pop[0] != 2 || pop[1] != 1 || pop[2] != 0 {
		t.Errorf("popularity incorrect after 3 requests: %v", pop)
	}

	// Record 4th request: 2 (window of 3: old [0] evicted, new window is 1, 0, 2)
	cache.RecordRequest(2)
	pop = cache.GetPopularityVector()
	if pop[0] != 1 || pop[1] != 1 || pop[2] != 1 {
		t.Errorf("popularity incorrect after window roll: %v", pop)
	}
}
