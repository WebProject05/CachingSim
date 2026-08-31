package core

import (
	"sync"

	"smdp-edge-caching-framework/pkg/config"
)

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

// CacheEngine manages edge router cache state, eviction, popularity history, and rewards.
type CacheEngine struct {
	mu            sync.RWMutex
	Cfg           *config.Config
	Files         []FileMetadata
	Cached        []bool // b(t), indexed by file ID
	UsedCapacity  float64
	RequestWindow []int // Sliding window ring buffer for N requests
	CurrentTime   float64
	sizes         []float64 // Immutable file sizes z_f
	popularity    []float64 // Incremental d(t) counts
	windowHead    int
	windowLen     int
	evictor       Evictor
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
	return &CacheEngine{
		Cfg:           cfg,
		Files:         CloneFiles(files),
		Cached:        make([]bool, n),
		UsedCapacity:  0.0,
		RequestWindow: make([]int, windowCap),
		CurrentTime:   0.0,
		sizes:         sizes,
		popularity:    make([]float64, n),
		evictor:       NewLowestUtilityEvictor(),
	}
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

// IsValidInsert checks if a file can theoretically fit within total cache capacity.
func (c *CacheEngine) IsValidInsert(fileID int) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if !c.validFileID(fileID) {
		return false
	}
	return c.Files[fileID].Size <= c.Cfg.CacheCapacity
}

// Insert adds a file to the cache if not already present.
func (c *CacheEngine) Insert(fileID int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.validFileID(fileID) || fileID >= len(c.Cached) || c.Cached[fileID] {
		return false
	}
	fileSize := c.Files[fileID].Size
	if (c.Cfg.CacheCapacity - c.UsedCapacity) < fileSize {
		return false
	}
	c.Cached[fileID] = true
	c.UsedCapacity += fileSize
	return true
}

// Remove deletes a file from the cache and updates used capacity.
func (c *CacheEngine) Remove(fileID int) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.validFileID(fileID) || fileID >= len(c.Cached) || !c.Cached[fileID] {
		return false
	}
	c.Cached[fileID] = false
	c.UsedCapacity -= c.Files[fileID].Size
	if c.UsedCapacity < 0 {
		c.UsedCapacity = 0
	}
	return true
}

// MemRatio computes the unoccupied proportion of cache memory Mem(t) (Section IV-A-1):
//
//	Mem(t) = (M - sum_{f=1}^F b(t) . z(t)) / M
func (c *CacheEngine) MemRatio() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.memRatioLocked()
}

func (c *CacheEngine) memRatioLocked() float64 {
	if c.Cfg.CacheCapacity <= 0 {
		return 0.0
	}
	ratio := (c.Cfg.CacheCapacity - c.UsedCapacity) / c.Cfg.CacheCapacity
	if ratio < 0.0 {
		return 0.0
	}
	if ratio > 1.0 {
		return 1.0
	}
	return ratio
}

// RecordRequest updates the sliding window popularity history d(t) and resets file generation time w_g^f.
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

	// Refresh file generation timestamp w_g^f = current time
	c.Files[fileID].GenTime = c.CurrentTime
}

// EvictUntilFits removes files with lowest utility until sufficient space is freed (Section IV-A-2).
func (c *CacheEngine) EvictUntilFits(newFileID int) []int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.validFileID(newFileID) || c.Cfg.CacheCapacity <= 0 {
		return nil
	}

	requiredSize := c.Files[newFileID].Size
	evicted, newUsed := c.evictor.Evict(
		c.Cached,
		c.Files,
		c.UsedCapacity,
		c.Cfg.CacheCapacity,
		requiredSize,
		c.CurrentTime,
		c.Cfg,
	)
	c.UsedCapacity = newUsed
	return evicted
}

// GetWorth computes the total worth of cached files W(t) (Eq. 3):
//
//	W(t) = (b(t) . d(t)) (b(t) . y(t))^T = sum_{f=1}^F b_f(t) * d_f(t) * y_f(t)
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
		u := c.Files[id].Utility(c.CurrentTime, c.Cfg)
		if id < len(c.popularity) {
			worth += c.popularity[id] * u
		}
	}
	return worth
}

// ComputeReward computes the instant reward r(t) (Eq. 2):
//
//	r(t) = W(t) - Mem(t) * 100
func (c *CacheEngine) ComputeReward() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	worth := c.getWorthLocked()
	mem := c.memRatioLocked()
	return worth - (mem * 100.0)
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
		y[i] = c.Files[i].Utility(c.CurrentTime, c.Cfg)
		if i < len(c.Cached) && c.Cached[i] {
			b[i] = 1
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

// Reset clears the cache state, popularity history, and resets simulation clock.
func (c *CacheEngine) Reset(files []FileMetadata) {
	c.mu.Lock()
	defer c.mu.Unlock()

	n := c.Cfg.TotalFileTypes
	if n < len(files) {
		n = len(files)
	}
	c.Files = CloneFiles(files)
	c.Cached = make([]bool, n)
	c.UsedCapacity = 0.0
	c.RequestWindow = make([]int, c.Cfg.SlidingWindowN)
	c.CurrentTime = 0.0
	c.sizes = make([]float64, n)
	for i := 0; i < len(files) && i < n; i++ {
		c.sizes[i] = files[i].Size
	}
	c.popularity = make([]float64, n)
	c.windowHead = 0
	c.windowLen = 0
}
