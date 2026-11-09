package core

import (
	"math"
	"sync"

	"smdp-edge-caching-framework/pkg/config"
)

// GhostEntry records metadata for rejected or evicted items tracked by Counterfactual Ghost Caching.
type GhostEntry struct {
	Timestamp     float64
	Cost          float64
	WasRejected   bool
	FromProtected bool
	WasExpired    bool
}

// State represents the system state s(t) at time t formulated in Section IV-A-1:
//
//	s(t) = {Mem(t), y(t), d(t), b(t), z(t), f_r}
type State struct {
	Mem           float64   // Unoccupied memory proportion: Mem(t) in [0, 1]
	D             []float64 // Frequency count of each file in recent N requests: d(t)
	Y             []float64 // Current utilities: y(t)
	Z             []float64 // File sizes: z(t) in MiB
	B             []int     // Binary cached status: b(t) in {0, 1}^F
	RequestedFile int       // Current requested file index f_r
}

// CacheEngine manages edge router cache state, eviction, popularity history,
// dynamic lifetimes, elastic capacity, two-tier probation shield, and counterfactual ghost memory.
type CacheEngine struct {
	mu            sync.RWMutex
	Cfg           *config.Config
	Files         []FileMetadata
	Cached        []bool // b(t), true if in Probation or Protected tier
	InProbation   []bool // true if file resides in Two-Tier Probationary Shield
	UsedCapacity  float64
	ProbationUsed float64
	ProtectedUsed float64
	RequestWindow []int // Sliding window ring buffer for N requests
	CurrentTime   float64
	sizes         []float64 // Immutable file sizes z_f
	popularity    []float64 // Incremental d(t) counts
	windowHead    int
	windowLen     int
	evictor       Evictor

	// Elastic scaling & Ghost Cache telemetry (AdaLife-Pro)
	TauEMA          float64
	LambdaEMA       float64
	RecentEvictions float64
	GhostHits       int
	Promotions      int
	LastStepRegret  float64
	ghostCache      map[int]GhostEntry
	ghostOrder      []int
}

// NewCacheEngine constructs a new CacheEngine instance.
func NewCacheEngine(cfg *config.Config, files []FileMetadata) *CacheEngine {
	n := cfg.TotalFileTypes
	if n < len(files) {
		n = len(files)
	}
	sizes := make([]float64, n)
	for i := 0; i < len(files) && i < n; i++ {
		sizes[i] = files[i].Size
	}
	windowCap := cfg.SlidingWindowN
	if windowCap < 0 {
		windowCap = 0
	}
	initLambda := 0.2
	if cfg.LambdaSource > 0 {
		initLambda = cfg.LambdaSource
	}
	initTau := 1.0 / initLambda
	return &CacheEngine{
		Cfg:           cfg,
		Files:         CloneFiles(files),
		Cached:        make([]bool, n),
		InProbation:   make([]bool, n),
		UsedCapacity:  0.0,
		RequestWindow: make([]int, windowCap),
		CurrentTime:   0.0,
		sizes:         sizes,
		popularity:    make([]float64, n),
		evictor:       evictorForConfig(cfg),
		TauEMA:        initTau,
		LambdaEMA:     initLambda,
		ghostCache:    make(map[int]GhostEntry),
	}
}

// evictorForConfig returns the configured evictor, defaulting to UtilityDensityEvictor (Resolving Gap #1).
func evictorForConfig(cfg *config.Config) Evictor {
	if cfg != nil && cfg.EvictionStrategy == "lowest_utility" {
		return NewLowestUtilityEvictor()
	}
	return NewUtilityDensityEvictor()
}

// SetEvictor allows swapping the eviction strategy dynamically.
func (c *CacheEngine) SetEvictor(e Evictor) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.evictor = e
}

// validFileID checks if the fileID is within valid bounds.
func (c *CacheEngine) validFileID(fileID int) bool {
	return fileID >= 0 && fileID < len(c.Files)
}

// IsCached returns whether the file is currently stored in the cache.
func (c *CacheEngine) IsCached(fileID int) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return fileID >= 0 && fileID < len(c.Cached) && c.Cached[fileID]
}

// EffectiveCapacity returns the dynamic elastic capacity M_eff(t) (AdaLife-Pro Layer 2).
func (c *CacheEngine) EffectiveCapacity() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.effectiveCapacityLocked()
}

func (c *CacheEngine) effectiveCapacityLocked() float64 {
	if c.Cfg == nil || c.Cfg.CacheCapacity <= 0 {
		return 0.0
	}
	if !c.Cfg.EnableElasticCapacity {
		return c.Cfg.CacheCapacity
	}
	// Only expand above base capacity under active traffic load (>0.5 req/s) or eviction pressure
	var pressure float64
	if c.LambdaEMA > 0.5 && c.Cfg.ElasticLambdaRef > 0 {
		pressure = math.Min(1.0, c.LambdaEMA/c.Cfg.ElasticLambdaRef)
	}
	if c.RecentEvictions > 0.1 {
		evictPressure := math.Min(1.0, c.RecentEvictions*0.25)
		if evictPressure > pressure {
			pressure = evictPressure
		}
	}
	return c.Cfg.CacheCapacity * (1.0 + c.Cfg.ElasticMaxScale*pressure)
}

// UpdateTrafficRate updates the unbiased Exponential Moving Average of inter-arrival time TauEMA
// and derives the smoothed arrival rate LambdaEMA = 1 / TauEMA.
func (c *CacheEngine) UpdateTrafficRate(tau float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if tau > 0 {
		if c.TauEMA <= 0 {
			c.TauEMA = tau
		} else {
			c.TauEMA = 0.05*tau + 0.95*c.TauEMA
		}
		c.LambdaEMA = 1.0 / math.Max(0.01, c.TauEMA)
	}
	c.RecentEvictions *= 0.98
}

// IsValidInsert checks if a file can theoretically fit within effective cache capacity.
func (c *CacheEngine) IsValidInsert(fileID int) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if !c.validFileID(fileID) {
		return false
	}
	return c.Files[fileID].Size <= c.effectiveCapacityLocked()
}

// Insert adds a file directly to the cache if not already present.
func (c *CacheEngine) Insert(fileID int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.validFileID(fileID) || fileID >= len(c.Cached) || c.Cached[fileID] {
		return false
	}
	fileSize := c.Files[fileID].Size
	effCap := c.effectiveCapacityLocked()
	if (effCap - c.UsedCapacity) < fileSize {
		return false
	}
	c.Cached[fileID] = true
	if fileID < len(c.InProbation) {
		c.InProbation[fileID] = false
	}
	c.Files[fileID].ResetLease(c.CurrentTime, 1.0)
	c.UsedCapacity += fileSize
	c.ProtectedUsed += fileSize
	return true
}

// IsFresh returns whether the file is currently stored in cache and has not expired.
func (c *CacheEngine) IsFresh(fileID int, currentTime float64) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.isFreshLocked(fileID, currentTime)
}

func (c *CacheEngine) isFreshLocked(fileID int, currentTime float64) bool {
	if !c.validFileID(fileID) || fileID >= len(c.Cached) || !c.Cached[fileID] {
		return false
	}
	return c.Files[fileID].Freshness(currentTime) < 1.0
}

// PurgeExpired removes all expired files (freshness >= 1.0) from both tiers and reclaims memory.
func (c *CacheEngine) PurgeExpired(currentTime float64) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.purgeExpiredLocked(currentTime)
}

func (c *CacheEngine) purgeExpiredLocked(currentTime float64) int {
	purged := 0
	for id := 0; id < len(c.Cached) && id < len(c.Files); id++ {
		if c.Cached[id] && c.Files[id].Freshness(currentTime) >= 1.0 {
			wasProtected := id >= len(c.InProbation) || !c.InProbation[id]
			c.removeLocked(id)
			c.recordGhostLocked(id, currentTime, false, wasProtected, true)
			purged++
		}
	}
	return purged
}

// CheckHit checks whether requested file is cached and valid at currentTime.
// If cached but expired (freshness >= 1.0), it purges the stale file and returns false.
func (c *CacheEngine) CheckHit(fileID int, currentTime float64) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.validFileID(fileID) || fileID >= len(c.Cached) || !c.Cached[fileID] {
		return false
	}
	if c.Files[fileID].Freshness(currentTime) < 1.0 {
		return true
	}
	// Expired: purge stale entry
	c.removeLocked(fileID)
	return false
}

// Remove deletes a file from whichever tier holds it and updates used capacity.
func (c *CacheEngine) Remove(fileID int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.removeLocked(fileID)
}

func (c *CacheEngine) removeLocked(fileID int) bool {
	if !c.validFileID(fileID) || fileID >= len(c.Cached) || !c.Cached[fileID] {
		return false
	}
	size := c.Files[fileID].Size
	c.Cached[fileID] = false
	if fileID < len(c.InProbation) && c.InProbation[fileID] {
		c.InProbation[fileID] = false
		c.ProbationUsed -= size
		if c.ProbationUsed < 0 {
			c.ProbationUsed = 0
		}
	} else {
		c.ProtectedUsed -= size
		if c.ProtectedUsed < 0 {
			c.ProtectedUsed = 0
		}
	}
	c.UsedCapacity -= size
	if c.UsedCapacity < 0 {
		c.UsedCapacity = 0
	}
	c.Files[fileID].HitCount = 0
	c.Files[fileID].EffectiveLifetime = 0
	return true
}

// MemRatio computes the unoccupied proportion of cache memory Mem(t) (Section IV-A-1):
//
//	Mem(t) = (M_eff(t) - sum_{f=1}^F b(t) . z(t)) / M_eff(t)
func (c *CacheEngine) MemRatio() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.memRatioLocked()
}

func (c *CacheEngine) memRatioLocked() float64 {
	effCap := c.effectiveCapacityLocked()
	if effCap <= 0 {
		return 0.0
	}
	ratio := (effCap - c.UsedCapacity) / effCap
	if ratio < 0.0 {
		return 0.0
	}
	if ratio > 1.0 {
		return 1.0
	}
	return ratio
}

// RecordRequest updates the sliding window popularity history d(t).
func (c *CacheEngine) RecordRequest(fileID int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.validFileID(fileID) {
		return
	}

	// Update sliding window ring buffer
	windowCap := len(c.RequestWindow)
	if windowCap > 0 {
		if c.windowLen == windowCap {
			old := c.RequestWindow[c.windowHead]
			if old >= 0 && old < len(c.popularity) {
				c.popularity[old]--
				if c.popularity[old] < 0 {
					c.popularity[old] = 0
				}
			}
			c.RequestWindow[c.windowHead] = fileID
			c.windowHead = (c.windowHead + 1) % windowCap
		} else {
			c.RequestWindow[c.windowLen] = fileID
			c.windowLen++
		}
		if fileID < len(c.popularity) {
			c.popularity[fileID]++
		}
	}
}

// EvictUntilFits removes files with lowest utility until sufficient space is freed (Section IV-A-2).
func (c *CacheEngine) EvictUntilFits(newFileID int) []int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.validFileID(newFileID) || c.Cfg.CacheCapacity <= 0 {
		return nil
	}

	requiredSize := c.Files[newFileID].Size
	effCap := c.effectiveCapacityLocked()
	evicted, newUsed := c.evictor.Evict(
		c.Cached,
		c.Files,
		c.UsedCapacity,
		effCap,
		requiredSize,
		c.CurrentTime,
		c.Cfg,
	)
	for _, evID := range evicted {
		wasProtected := evID >= len(c.InProbation) || !c.InProbation[evID]
		c.removeBookkeepingLocked(evID)
		c.recordGhostLocked(evID, c.CurrentTime, false, wasProtected, false)
	}
	c.UsedCapacity = newUsed
	c.RecentEvictions += float64(len(evicted))
	return evicted
}

// removeBookkeepingLocked updates tier accounting and resets file metadata on removal.
func (c *CacheEngine) removeBookkeepingLocked(fileID int) {
	if !c.validFileID(fileID) {
		return
	}
	size := c.Files[fileID].Size
	if fileID < len(c.InProbation) && c.InProbation[fileID] {
		c.InProbation[fileID] = false
		c.ProbationUsed -= size
		if c.ProbationUsed < 0 {
			c.ProbationUsed = 0
		}
	} else {
		c.ProtectedUsed -= size
		if c.ProtectedUsed < 0 {
			c.ProtectedUsed = 0
		}
	}
	c.Files[fileID].HitCount = 0
	c.Files[fileID].EffectiveLifetime = 0
}

// recordGhostLocked records a rejected or evicted file ID into the bounded Shadow Ghost Cache.
func (c *CacheEngine) recordGhostLocked(fileID int, currentTime float64, wasRejected, fromProtected, wasExpired bool) {
	if c.Cfg == nil || !c.Cfg.EnableGhostCache || !c.validFileID(fileID) {
		return
	}
	if c.ghostCache == nil {
		c.ghostCache = make(map[int]GhostEntry)
	}
	maxEntries := c.Cfg.GhostMaxEntries
	if maxEntries <= 0 {
		maxEntries = 1000
	}
	if _, exists := c.ghostCache[fileID]; !exists {
		if len(c.ghostOrder) >= maxEntries {
			oldest := c.ghostOrder[0]
			c.ghostOrder = c.ghostOrder[1:]
			delete(c.ghostCache, oldest)
		}
		c.ghostOrder = append(c.ghostOrder, fileID)
	}
	c.ghostCache[fileID] = GhostEntry{
		Timestamp:     currentTime,
		Cost:          c.Files[fileID].FetchCost(c.Cfg),
		WasRejected:   wasRejected,
		FromProtected: fromProtected,
		WasExpired:    wasExpired,
	}
}

// PeekGhost checks whether fileID currently exists in Counterfactual Ghost Cache within GhostHorizon.
func (c *CacheEngine) PeekGhost(fileID int, currentTime float64) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if c.Cfg == nil || !c.Cfg.EnableGhostCache || c.ghostCache == nil {
		return false
	}
	entry, ok := c.ghostCache[fileID]
	if !ok {
		return false
	}
	horizon := c.Cfg.GhostHorizon
	if horizon <= 0 {
		horizon = 60.0
	}
	return (currentTime - entry.Timestamp) <= horizon
}

// TierCapacities returns the Probation and Protected tier capacities for a candidate of reqFileSize.
func (c *CacheEngine) TierCapacities(reqFileSize float64) (probCap, protCap float64) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tierCapacitiesLocked(reqFileSize)
}

// consumeGhostHitLocked checks if a missed file was recently evicted or rejected (Counterfactual Ghost Hit).
func (c *CacheEngine) consumeGhostHitLocked(fileID int, currentTime float64) bool {
	c.LastStepRegret = 0.0
	if c.Cfg == nil || !c.Cfg.EnableGhostCache || c.ghostCache == nil {
		return false
	}
	entry, ok := c.ghostCache[fileID]
	if !ok {
		return false
	}
	delete(c.ghostCache, fileID)
	horizon := c.Cfg.GhostHorizon
	if horizon <= 0 {
		horizon = 60.0
	}
	if (currentTime - entry.Timestamp) <= horizon {
		c.GhostHits++
		u := c.Cfg.UTMax * c.Files[fileID].Importance
		if entry.WasExpired {
			c.LastStepRegret = 0.0
		} else if entry.WasRejected || entry.FromProtected {
			c.LastStepRegret = c.Cfg.GhostRegretWeight * u * entry.Cost
		} else {
			c.LastStepRegret = 0.25 * c.Cfg.GhostRegretWeight * u * entry.Cost
		}
		return true
	}
	return false
}

// tierCapacitiesLocked calculates the capacity budget for Probation and Protected tiers.
func (c *CacheEngine) tierCapacitiesLocked(reqFileSize float64) (probCap, protCap float64) {
	effCap := c.effectiveCapacityLocked()
	if c.Cfg == nil || !c.Cfg.EnableProbationShield || effCap <= 0 {
		return 0, effCap
	}
	ratio := c.Cfg.ProbationRatio
	if ratio <= 0 || ratio >= 1 {
		ratio = 0.15
	}
	probCap = effCap * ratio
	// Ensure probation can hold at least one incoming file when total cache allows it
	if reqFileSize > probCap && effCap >= reqFileSize*2.0 {
		probCap = math.Min(effCap*0.25, reqFileSize*1.05)
	}
	protCap = effCap - probCap
	return probCap, protCap
}

// evictProbationLocked evicts oldest entries exclusively from Probation without touching Protected tier.
func (c *CacheEngine) evictProbationLocked(needed float64, currentTime float64) int {
	evictions := 0
	for needed > 0 {
		oldestID := -1
		oldestTime := math.MaxFloat64
		for id := 0; id < len(c.Cached) && id < len(c.Files); id++ {
			if c.Cached[id] && id < len(c.InProbation) && c.InProbation[id] {
				if c.Files[id].GenTime < oldestTime {
					oldestTime = c.Files[id].GenTime
					oldestID = id
				}
			}
		}
		if oldestID == -1 {
			break
		}
		size := c.Files[oldestID].Size
		c.removeLocked(oldestID)
		c.recordGhostLocked(oldestID, currentTime, false, false, false)
		needed -= size
		evictions++
	}
	return evictions
}

// evictProtectedLocked evicts lowest Cost-Utility Density entries from the Protected tier.
func (c *CacheEngine) evictProtectedLocked(needed float64, currentTime float64, trackPressure bool) int {
	evictions := 0
	for needed > 0 {
		worstID := -1
		minDensity := math.MaxFloat64
		for id := 0; id < len(c.Cached) && id < len(c.Files); id++ {
			if c.Cached[id] && (id >= len(c.InProbation) || !c.InProbation[id]) {
				u := c.Files[id].Utility(currentTime, c.Cfg) * c.Files[id].FetchCost(c.Cfg)
				sz := math.Max(1.0, c.Files[id].Size)
				density := u / sz
				if density < minDensity {
					minDensity = density
					worstID = id
				}
			}
		}
		if worstID == -1 {
			break
		}
		size := c.Files[worstID].Size
		c.removeLocked(worstID)
		c.recordGhostLocked(worstID, currentTime, false, true, false)
		needed -= size
		evictions++
	}
	if trackPressure {
		c.RecentEvictions += float64(evictions)
	}
	return evictions
}

// ProcessAdaLifeStep executes one full cache transition with all 5 AdaLife-Pro mechanisms enabled:
// 1. Dynamic Lifetime lease extensions on hit
// 2. Elastic Capacity headroom during surges
// 3. Two-Tier Probationary Shield filtering one-hit wonders
// 4. Fetch-Cost & Latency Asymmetry in cost-utility density
// 5. Counterfactual Ghost Cache regret feedback and fast-track promotion
func (c *CacheEngine) ProcessAdaLifeStep(reqFile int, action int, currentTime float64) (isHit bool, utilityGained float64, evictions int, inserted bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.CurrentTime = currentTime
	c.LastStepRegret = 0.0
	c.purgeExpiredLocked(currentTime)

	if !c.validFileID(reqFile) {
		return false, 0.0, 0, false
	}

	fileSize := c.Files[reqFile].Size
	effCap := c.effectiveCapacityLocked()

	// Gracefully trim surplus capacity if Elastic Capacity has scaled down during quieter traffic
	if c.UsedCapacity > effCap && effCap > 0 {
		probCap, _ := c.tierCapacitiesLocked(fileSize)
		if c.Cfg.EnableProbationShield && c.ProbationUsed > probCap {
			evictions += c.evictProbationLocked(c.ProbationUsed-probCap, currentTime)
		}
		if c.UsedCapacity > effCap {
			evictions += c.evictProtectedLocked(c.UsedCapacity-effCap, currentTime, false)
		}
	}

	// Check Cache Hit (Protected or Probation)
	if reqFile < len(c.Cached) && c.Cached[reqFile] && c.Files[reqFile].Freshness(currentTime) < 1.0 {
		isHit = true
		utilityGained = c.Files[reqFile].Utility(currentTime, c.Cfg)

		// If hit occurs in Probationary Tier -> PROMOTE to Protected Main Tier!
		if c.Cfg.EnableProbationShield && reqFile < len(c.InProbation) && c.InProbation[reqFile] {
			_, protCap := c.tierCapacitiesLocked(fileSize)
			if protCap >= fileSize {
				// Leave probation cleanly
				c.InProbation[reqFile] = false
				c.ProbationUsed -= fileSize
				if c.ProbationUsed < 0 {
					c.ProbationUsed = 0
				}
				c.UsedCapacity -= fileSize

				// Free space in Protected Tier only if full
				if (protCap - c.ProtectedUsed) < fileSize {
					needed := fileSize - (protCap - c.ProtectedUsed)
					evictions += c.evictProtectedLocked(needed, currentTime, true)
				}
				if (protCap - c.ProtectedUsed) >= fileSize {
					c.ProtectedUsed += fileSize
					c.UsedCapacity += fileSize
					c.Files[reqFile].ResetLease(currentTime, 1.0)
					c.Files[reqFile].RenewLease(currentTime, c.Cfg)
					c.Promotions++
				} else {
					// Revert to probation if protected cannot fit
					c.InProbation[reqFile] = true
					c.ProbationUsed += fileSize
					c.UsedCapacity += fileSize
				}
			}
		} else {
			// Hit directly in Protected Main Tier -> Dynamic Lease Extension!
			c.Files[reqFile].RenewLease(currentTime, c.Cfg)
		}
		return isHit, utilityGained, evictions, false
	}

	// Cache Miss: Check Counterfactual Ghost Cache
	isGhostHit := c.consumeGhostHitLocked(reqFile, currentTime)

	if fileSize > effCap {
		return false, 0.0, evictions, false
	}

	if action == 0 {
		if isGhostHit && c.Cfg.EnableGhostCache {
			u := c.Cfg.UTMax * c.Files[reqFile].Importance
			c.LastStepRegret += 0.5 * c.Cfg.GhostRegretWeight * u * c.Files[reqFile].FetchCost(c.Cfg)
		}
		c.recordGhostLocked(reqFile, currentTime, true, false, false)
		return false, 0.0, evictions, false
	}

	// Action == 1: Admit candidate according to Probationary Shield / Fast-Track rules
	if !c.Cfg.EnableProbationShield {
		if (effCap - c.UsedCapacity) < fileSize {
			needed := fileSize - (effCap - c.UsedCapacity)
			evictions += c.evictProtectedLocked(needed, currentTime, true)
		}
		if (effCap - c.UsedCapacity) >= fileSize {
			c.Cached[reqFile] = true
			c.InProbation[reqFile] = false
			c.Files[reqFile].ResetLease(currentTime, 1.0)
			c.ProtectedUsed += fileSize
			c.UsedCapacity += fileSize
			inserted = true
		}
		return false, 0.0, evictions, inserted
	}

	probCap, protCap := c.tierCapacitiesLocked(fileSize)

	// Fast-track directly into Protected Tier on a Ghost Hit OR proven returning item (popularity >= 2 in recent window)
	isReturningPopular := isGhostHit || (reqFile < len(c.popularity) && c.popularity[reqFile] >= 2.0)
	if isReturningPopular && protCap >= fileSize {
		if (protCap - c.ProtectedUsed) < fileSize {
			needed := fileSize - (protCap - c.ProtectedUsed)
			evictions += c.evictProtectedLocked(needed, currentTime, true)
		}
		if (effCap - c.UsedCapacity) < fileSize {
			neededTotal := fileSize - (effCap - c.UsedCapacity)
			if c.ProbationUsed > 0 {
				evictions += c.evictProbationLocked(neededTotal, currentTime)
			}
			if (effCap - c.UsedCapacity) < fileSize {
				evictions += c.evictProtectedLocked(fileSize-(effCap-c.UsedCapacity), currentTime, false)
			}
		}
		if (protCap-c.ProtectedUsed) >= fileSize && (effCap-c.UsedCapacity) >= fileSize {
			c.Cached[reqFile] = true
			c.InProbation[reqFile] = false
			c.Files[reqFile].ResetLease(currentTime, 1.0)
			if isGhostHit || (reqFile < len(c.popularity) && c.popularity[reqFile] >= 3.0) {
				c.Files[reqFile].RenewLease(currentTime, c.Cfg)
			}
			c.ProtectedUsed += fileSize
			c.UsedCapacity += fileSize
			c.Promotions++
			inserted = true
			return false, 0.0, evictions, inserted
		}
	}

	// Normal miss -> Admit into Probationary Tier.
	// Allow Probation to utilize unoccupied idle RAM while Protected warms up,
	// clamping down strictly to probCap once Protected fills protCap (without evicting Protected items).
	probMaxBudget := math.Max(probCap, effCap-c.ProtectedUsed)
	if probMaxBudget >= fileSize {
		if (effCap - c.UsedCapacity) < fileSize {
			needed := fileSize - (effCap - c.UsedCapacity)
			if c.ProbationUsed > 0 {
				evictions += c.evictProbationLocked(needed, currentTime)
			}
			// Only if cache is still tight (e.g., oversized Protected tier overflowed protCap), trim excess Protected
			if (effCap - c.UsedCapacity) < fileSize {
				neededProt := fileSize - (effCap - c.UsedCapacity)
				evictions += c.evictProtectedLocked(neededProt, currentTime, false)
			}
		}
		if (effCap - c.UsedCapacity) >= fileSize {
			ttlScale := c.Cfg.ProbationTTLScale
			if ttlScale <= 0 {
				ttlScale = 0.75
			}
			c.Cached[reqFile] = true
			c.InProbation[reqFile] = true
			c.Files[reqFile].ResetLease(currentTime, ttlScale)
			c.ProbationUsed += fileSize
			c.UsedCapacity += fileSize
			inserted = true
			return false, 0.0, evictions, inserted
		}
	}

	// File too big for probation on 1st miss -> record in Ghost so 2nd request fast-tracks
	c.recordGhostLocked(reqFile, currentTime, false, false, false)
	return false, 0.0, evictions, false
}

// GetWorth computes the total worth of cached files W(t) (Eq. 3):
// When EnableFetchCost is active, scales each file's worth by its origin retrieval cost C_fetch.
func (c *CacheEngine) GetWorth() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.getWorthLocked()
}

func (c *CacheEngine) getWorthLocked() float64 {
	var worth float64
	for id := 0; id < len(c.Cached) && id < len(c.Files); id++ {
		if !c.Cached[id] {
			continue
		}
		u := c.Files[id].Utility(c.CurrentTime, c.Cfg) * c.Files[id].FetchCost(c.Cfg)
		if id < len(c.popularity) {
			worth += c.popularity[id] * u
		}
	}
	return worth
}

// ComputeReward computes the instant reward r(t) (Eq. 2 & Enhanced AdaLife-Pro Reward):
// r(t) = W*(t) - Mem(t) * w_mem - CounterfactualRegret(t)
func (c *CacheEngine) ComputeReward() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	worth := c.getWorthLocked()
	mem := c.memRatioLocked()
	penaltyWeight := 10.0
	if c.Cfg != nil && c.Cfg.MemPenaltyWeight > 0 {
		penaltyWeight = c.Cfg.MemPenaltyWeight
	}
	regret := 0.0
	if c.Cfg != nil && c.Cfg.EnableGhostCache {
		regret = c.LastStepRegret
	}
	return worth - (mem * penaltyWeight) - regret
}

// GetPopularityVector returns a copy of the popularity vector d(t).
func (c *CacheEngine) GetPopularityVector() []float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	d := make([]float64, c.Cfg.TotalFileTypes)
	copy(d, c.popularity)
	return d
}

// GetCurrentState constructs the complete state vector s(t) (Section IV-A-1).
func (c *CacheEngine) GetCurrentState(requestedFile int) *State {
	c.mu.RLock()
	defer c.mu.RUnlock()

	F := c.Cfg.TotalFileTypes
	d := make([]float64, F)
	y := make([]float64, F)
	z := make([]float64, F)
	b := make([]int, F)

	copy(d, c.popularity)
	copy(z, c.sizes)

	limit := F
	if limit > len(c.Files) {
		limit = len(c.Files)
	}

	for i := 0; i < limit; i++ {
		if i < len(c.Cached) && c.Cached[i] && c.isFreshLocked(i, c.CurrentTime) {
			y[i] = c.Files[i].Utility(c.CurrentTime, c.Cfg) * c.Files[i].FetchCost(c.Cfg)
			b[i] = 1
		} else if i == requestedFile {
			// Newly fetched candidate from data center is brand new (freshness = 0)
			y[i] = c.Cfg.UTMax * c.Files[i].Importance * c.Files[i].FetchCost(c.Cfg)
		} else {
			y[i] = c.Files[i].Utility(c.CurrentTime, c.Cfg) * c.Files[i].FetchCost(c.Cfg)
		}
	}

	return &State{
		Mem:           c.memRatioLocked(),
		D:             d,
		Y:             y,
		Z:             z,
		B:             b,
		RequestedFile: requestedFile,
	}
}

// Reset clears the cache state, popularity history, tiers, ghost memory, and resets simulation clock.
func (c *CacheEngine) Reset(files []FileMetadata) {
	c.mu.Lock()
	defer c.mu.Unlock()

	n := c.Cfg.TotalFileTypes
	if n < len(files) {
		n = len(files)
	}
	c.Files = CloneFiles(files)
	c.Cached = make([]bool, n)
	c.InProbation = make([]bool, n)
	c.UsedCapacity = 0.0
	c.ProbationUsed = 0.0
	c.ProtectedUsed = 0.0
	c.RequestWindow = make([]int, c.Cfg.SlidingWindowN)
	c.CurrentTime = 0.0
	c.sizes = make([]float64, n)
	for i := 0; i < len(files) && i < n; i++ {
		c.sizes[i] = files[i].Size
	}
	c.popularity = make([]float64, n)
	c.windowHead = 0
	c.windowLen = 0
	initLambda := 0.2
	if c.Cfg.LambdaSource > 0 {
		initLambda = c.Cfg.LambdaSource
	}
	c.TauEMA = 1.0 / initLambda
	c.LambdaEMA = initLambda
	c.RecentEvictions = 0.0
	c.GhostHits = 0
	c.Promotions = 0
	c.LastStepRegret = 0.0
	c.ghostCache = make(map[int]GhostEntry)
	c.ghostOrder = nil
}
