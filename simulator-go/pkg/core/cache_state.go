package core

// Manages the cache memory, eviction logic, and constructs the State Vector
import (
	"math"
	"smdp-edge-caching-framework/pkg/config"
)

type State struct {
	Mem           float64   // Unoccupied memory proportion: 1 - sum(b_f * z_f)/M
	D             []float64 // Frequency count in last N requests: d(t)
	Y             []float64 // Current utilities: y(t)
	Z             []float64 // File sizes: z(t)
	B             []int     // Binary cached status: b(t) in {0, 1}^F
	RequestedFile int       // Current requested file index f_r
}

type CacheEngine struct {
	Cfg           *config.Config
	Files         []FileMetadata
	Cached        []bool // b(t), indexed by file ID
	UsedCapacity  float64
	RequestWindow []int // Sliding window ring for N requests
	CurrentTime   float64
	sizes         []float64 // Immutable file sizes z_f
	popularity    []float64 // Incremental d(t)
	windowHead    int
	windowLen     int
}

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
		Files:         files,
		Cached:        make([]bool, n),
		UsedCapacity:  0.0,
		RequestWindow: make([]int, windowCap),
		CurrentTime:   0.0,
		sizes:         sizes,
		popularity:    make([]float64, n),
	}
}

func (c *CacheEngine) validFileID(fileID int) bool {
	return fileID >= 0 && fileID < len(c.Files)
}

func (c *CacheEngine) IsCached(fileID int) bool {
	return fileID >= 0 && fileID < len(c.Cached) && c.Cached[fileID]
}

func (c *CacheEngine) IsValidInsert(fileID int) bool {
	return c.validFileID(fileID)
}

func (c *CacheEngine) Insert(fileID int) {
	if !c.validFileID(fileID) || fileID >= len(c.Cached) || c.Cached[fileID] {
		return
	}
	c.Cached[fileID] = true
	c.UsedCapacity += c.Files[fileID].Size
}

func (c *CacheEngine) memRatio() float64 {
	if c.Cfg.CacheCapacity <= 0 {
		return 0
	}
	ratio := (c.Cfg.CacheCapacity - c.UsedCapacity) / c.Cfg.CacheCapacity
	if ratio < 0.0 {
		return 0.0
	}
	return ratio
}

// RecordRequest updates sliding window d(t) and refreshes file generation time
func (c *CacheEngine) RecordRequest(fileID int) {
	if !c.validFileID(fileID) || c.Cfg.SlidingWindowN <= 0 || len(c.RequestWindow) == 0 {
		return
	}
	n := len(c.RequestWindow)
	if c.windowLen == n {
		old := c.RequestWindow[c.windowHead]
		if old >= 0 && old < len(c.popularity) {
			c.popularity[old]--
		}
		c.RequestWindow[c.windowHead] = fileID
		c.windowHead++
		if c.windowHead == n {
			c.windowHead = 0
		}
	} else {
		c.RequestWindow[c.windowLen] = fileID
		c.windowLen++
	}
	if fileID < len(c.popularity) {
		c.popularity[fileID]++
	}
	c.Files[fileID].GenTime = c.CurrentTime
}

// EvictUntilFits removes the file with lowest utility until new file has the space to occupy in the cache
func (c *CacheEngine) EvictUntilFits(newFileID int) {
	if !c.validFileID(newFileID) || c.Cfg.CacheCapacity <= 0 {
		return
	}
	requiredSize := c.Files[newFileID].Size
	for (c.Cfg.CacheCapacity - c.UsedCapacity) < requiredSize {
		lowestID := -1
		minUtility := math.MaxFloat64

		for id := 0; id < len(c.Cached); id++ {
			if !c.Cached[id] || id >= len(c.Files) {
				continue
			}
			u := c.Files[id].Utility(c.CurrentTime, c.Cfg)
			if u < minUtility {
				minUtility = u
				lowestID = id
			}
		}

		if lowestID == -1 {
			break
		}

		c.Cached[lowestID] = false
		c.UsedCapacity -= c.Files[lowestID].Size
		if c.UsedCapacity < 0 {
			c.UsedCapacity = 0
		}
	}
}

// ComputeReward calculates instant reward r(t) "Eq(2) and Eq(3)""
func (c *CacheEngine) ComputeReward() float64 {
	if c.Cfg.CacheCapacity <= 0 {
		return 0
	}

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

	return worth - (c.memRatio() * 100.0)
}

func (c *CacheEngine) GetPopularityVector() []float64 {
	d := make([]float64, len(c.popularity))
	copy(d, c.popularity)
	if extra := c.Cfg.TotalFileTypes - len(d); extra > 0 {
		d = append(d, make([]float64, extra)...)
	}
	if c.Cfg.TotalFileTypes >= 0 && c.Cfg.TotalFileTypes < len(d) {
		d = d[:c.Cfg.TotalFileTypes]
	}
	return d
}

// GetCurrentState builds s(t) = {Mem(t), d(t), y(t), z(t), b(t)}
func (c *CacheEngine) GetCurrentState(requestedFile int) *State {
	F := c.Cfg.TotalFileTypes
	if F < 0 {
		F = 0
	}

	d := make([]float64, F)
	y := make([]float64, F)
	z := make([]float64, F)
	b := make([]int, F)

	limit := F
	if limit > len(c.Files) {
		limit = len(c.Files)
	}
	copy(d, c.popularity)
	copy(z, c.sizes)
	for i := 0; i < limit; i++ {
		y[i] = c.Files[i].Utility(c.CurrentTime, c.Cfg)
		if i < len(c.Cached) && c.Cached[i] {
			b[i] = 1
		}
	}

	return &State{
		Mem:           c.memRatio(),
		D:             d,
		Y:             y,
		Z:             z,
		B:             b,
		RequestedFile: requestedFile,
	}
}
