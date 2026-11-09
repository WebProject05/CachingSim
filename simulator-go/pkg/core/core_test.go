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

func TestDynamicLifetimeLeaseExtension(t *testing.T) {
	cfg := config.DefaultConfig()
	file := &FileMetadata{
		ID:         0,
		Size:       200.0,
		Lifetime:   20.0,
		Importance: 0.8,
		GenTime:    0.0,
	}

	file.ResetLease(10.0, 1.0)
	if file.ActiveLifetime() != 20.0 || file.HitCount != 1 {
		t.Fatalf("unexpected initial lease: ttl=%f, hits=%d", file.ActiveLifetime(), file.HitCount)
	}

	// Hit at t=15 -> boost = 1 + 0.5*0.8 + 0.1*2 = 1.6 -> EffectiveLifetime = 32.0
	file.RenewLease(15.0, cfg)
	expected := 20.0 * (1.0 + 0.5*0.8 + 0.1*2.0)
	if math.Abs(file.ActiveLifetime()-expected) > 1e-6 {
		t.Fatalf("expected renewed TTL %f, got %f", expected, file.ActiveLifetime())
	}
	if file.GenTime != 15.0 || file.HitCount != 2 {
		t.Fatalf("expected GenTime=15.0 and HitCount=2, got %f, %d", file.GenTime, file.HitCount)
	}

	// Verify MaxLifetimeCap bound
	cfg.MaxLifetimeCap = 35.0
	for i := 0; i < 6; i++ {
		file.RenewLease(20.0+float64(i), cfg)
	}
	if file.ActiveLifetime() > 35.0 {
		t.Fatalf("lease exceeded MaxLifetimeCap: got %f > 35.0", file.ActiveLifetime())
	}

	// Verify disabled AdaLife leaves base Lifetime untouched
	cfg.EnableAdaLife = false
	beforeTTL := file.EffectiveLifetime
	file.RenewLease(50.0, cfg)
	if file.EffectiveLifetime != beforeTTL {
		t.Fatalf("expected RenewLease to be no-op when EnableAdaLife=false")
	}
}

func TestElasticCapacityScaling(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.CacheCapacity = 1000.0
	cfg.LambdaSource = 0.2
	cfg.EnableElasticCapacity = true
	cfg.ElasticMaxScale = 0.35
	cfg.ElasticLambdaRef = 5.0

	files := []FileMetadata{{ID: 0, Size: 200.0, Lifetime: 20.0, Importance: 0.5}}
	cache := NewCacheEngine(cfg, files)

	// Idle traffic (lambda=0.2 <= 0.5 threshold) should equal exact base capacity
	if math.Abs(cache.EffectiveCapacity()-1000.0) > 1e-6 {
		t.Fatalf("expected idle effective capacity 1000.0, got %f", cache.EffectiveCapacity())
	}

	// Simulate sustained burst arrival rate (tau = 0.2s -> lambda = 5.0 req/s)
	for i := 0; i < 150; i++ {
		cache.UpdateTrafficRate(0.2)
	}
	effCap := cache.EffectiveCapacity()
	if effCap < 1340.0 || effCap > 1351.0 {
		t.Fatalf("expected elastic capacity near 1350.0 under 5 req/s burst, got %f (lambdaEMA=%f)", effCap, cache.LambdaEMA)
	}

	// Disabled flag should remain strictly 1000.0
	cfg.EnableElasticCapacity = false
	if cache.EffectiveCapacity() != 1000.0 {
		t.Fatalf("expected disabled elastic capacity 1000.0, got %f", cache.EffectiveCapacity())
	}
}

func TestProbationShieldAndGhostFastTrack(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TotalFileTypes = 6
	cfg.CacheCapacity = 1000.0
	cfg.EnableElasticCapacity = false // Isolate Probation + Ghost mechanics clearly
	cfg.ProbationRatio = 0.20         // 200 MiB Probation, 800 MiB Protected

	files := []FileMetadata{
		{ID: 0, Size: 800.0, Lifetime: 20.0, Importance: 0.9}, // Fills full 800 MiB Protected tier
		{ID: 1, Size: 200.0, Lifetime: 20.0, Importance: 0.2}, // One-hit noise #1
		{ID: 2, Size: 200.0, Lifetime: 20.0, Importance: 0.2}, // One-hit noise #2
		{ID: 3, Size: 200.0, Lifetime: 20.0, Importance: 0.2}, // One-hit noise #3
		{ID: 4, Size: 200.0, Lifetime: 20.0, Importance: 0.8}, // Rejected then Ghost-hit candidate
	}
	cache := NewCacheEngine(cfg, files)

	// 1. Place hot File 0 into Protected directly via 2-step promotion or ghost promotion
	cache.ProcessAdaLifeStep(0, 1, 1.0) // First miss (fits into idle RAM in Probation)
	if !cache.IsCached(0) || !cache.InProbation[0] {
		t.Fatalf("expected file 0 in probation on 1st miss")
	}
	// 2. Second request hits and promotes File 0 (800 MiB) into Protected Main Tier (fills all 800 MiB)
	hit, _, _, _ := cache.ProcessAdaLifeStep(0, 1, 2.0)
	if !hit || cache.InProbation[0] || !cache.IsCached(0) || cache.Promotions != 1 {
		t.Fatalf("expected file 0 to promote to protected on 2nd request (hit=%t prob=%t prom=%d)", hit, cache.InProbation[0], cache.Promotions)
	}

	// 3. Burst of one-hit wonders (Files 1, 2, 3) rotates strictly inside 200 MiB Probation without disturbing Protected File 0
	cache.ProcessAdaLifeStep(1, 1, 3.0)
	cache.ProcessAdaLifeStep(2, 1, 4.0)
	cache.ProcessAdaLifeStep(3, 1, 5.0)
	if !cache.IsCached(0) || cache.InProbation[0] {
		t.Fatalf("probation noise displaced or mutated protected file 0")
	}
	if cache.IsCached(1) || cache.IsCached(2) || !cache.IsCached(3) {
		t.Fatalf("expected probation FIFO shield to retain latest file 3 and isolate noise 1,2 (1=%t 2=%t 3=%t)",
			cache.IsCached(1), cache.IsCached(2), cache.IsCached(3))
	}

	// 4. Reject File 4 (action=0) -> logs into Ghost Cache; 2nd arrival incurs regret & fast-tracks to Protected!
	cache.ProcessAdaLifeStep(4, 0, 6.0)
	if !cache.PeekGhost(4, 7.0) {
		t.Fatalf("expected rejected file 4 to be present in Ghost Cache")
	}
	cache.RecordRequest(0)
	cache.RecordRequest(4)
	baseRewardWithoutRegret := cache.GetWorth() - cache.MemRatio()*cfg.MemPenaltyWeight

	hit, _, _, ins := cache.ProcessAdaLifeStep(4, 1, 10.0)
	if hit || !ins || cache.InProbation[4] || cache.GhostHits != 1 {
		t.Fatalf("expected returning ghost file 4 to fast-track straight to Protected (hit=%t ins=%t prob=%t ghosts=%d)",
			hit, ins, cache.InProbation[4], cache.GhostHits)
	}
	if cache.LastStepRegret <= 0.0 {
		t.Fatalf("expected positive LastStepRegret on avoidable ghost miss, got %f", cache.LastStepRegret)
	}
	rewardWithRegret := cache.ComputeReward()
	netWorthMinusMem := cache.GetWorth() - cache.MemRatio()*cfg.MemPenaltyWeight
	if math.Abs((netWorthMinusMem-rewardWithRegret)-cache.LastStepRegret) > 1e-6 {
		t.Fatalf("ComputeReward did not subtract LastStepRegret accurately (expected diff %f, got %f)",
			cache.LastStepRegret, netWorthMinusMem-rewardWithRegret)
	}
	_ = baseRewardWithoutRegret
}

func TestFetchCostAsymmetryEviction(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.CacheCapacity = 1000.0
	cfg.EnableElasticCapacity = false
	cfg.EnableFetchCost = true

	// File 0: Size=400, Importance=0.5 -> Utility(0)=0.75, FetchCost=0.8+0.16=0.96 -> CostUtilDensity = 0.720/400 = 0.001800
	// File 1: Size=600, Importance=0.5 -> Utility(0)=0.75, FetchCost=0.8+0.24=1.04 -> CostUtilDensity = 0.780/600 = 0.001300
	files := []FileMetadata{
		{ID: 0, Size: 400.0, Lifetime: 20.0, Importance: 0.5, GenTime: 0.0},
		{ID: 1, Size: 600.0, Lifetime: 20.0, Importance: 0.5, GenTime: 0.0},
		{ID: 2, Size: 500.0, Lifetime: 20.0, Importance: 0.9, GenTime: 0.0},
	}
	cache := NewCacheEngine(cfg, files)
	cache.Insert(0)
	cache.Insert(1)

	evicted := cache.EvictUntilFits(2)
	if len(evicted) != 1 || evicted[0] != 1 {
		t.Fatalf("expected lower cost-utility density file 1 to be evicted first, got %v", evicted)
	}
}
