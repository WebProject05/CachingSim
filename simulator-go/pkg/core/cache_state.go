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
	Cached        map[int]bool // b(t)
	UsedCapacity  float64
	RequestWindow []int   // Sliding window for N requests
	CurrentTime   float64 // Continuous physical simulation time t
}

func NewCacheEngine(cfg *config.Config, files []FileMetadata) *CacheEngine {
	return &CacheEngine{
		Cfg:           cfg,
		Files:         files,
		Cached:        make(map[int]bool),
		UsedCapacity:  0.0,
		RequestWindow: make([]int, 0, cfg.SlidingWindowN),
		CurrentTime:   0.0,
	}
}

func (c *CacheEngine) validFileID(fileID int) bool {
	return fileID >= 0 && fileID < len(c.Files)
}

// RecordRequest updates sliding window d(t) and refreshes file generation time
func (c *CacheEngine) RecordRequest(fileID int) {
	if !c.validFileID(fileID) || c.Cfg.SlidingWindowN <= 0 {
		return
	}
	if len(c.RequestWindow) >= c.Cfg.SlidingWindowN {
		c.RequestWindow = c.RequestWindow[1:]
	}
	c.RequestWindow = append(c.RequestWindow, fileID)
	// Refresh file generation timestamp upon request arrival
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

		for id := range c.Cached {
			u := c.Files[id].Utility(c.CurrentTime, c.Cfg)
			if u < minUtility {
				minUtility = u
				lowestID = id
			}
		}

		if lowestID == -1 {
			break
		}

		// Evict file with lowest utility
		delete(c.Cached, lowestID)
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
	memRatio := (c.Cfg.CacheCapacity - c.UsedCapacity) / c.Cfg.CacheCapacity
	if memRatio < 0.0 {
		memRatio = 0.0
	}

	d := c.GetPopularityVector()
	var worth float64
	for id := range c.Cached {
		u := c.Files[id].Utility(c.CurrentTime, c.Cfg)
		worth += d[id] * u
	}

	return worth - (memRatio * 100.0)
}

func (c *CacheEngine) GetPopularityVector() []float64 {
	d := make([]float64, c.Cfg.TotalFileTypes)
	for _, id := range c.RequestWindow {
		if id >= 0 && id < len(d) {
			d[id]++
		}
	}
	return d
}

// GetCurrentState builds s(t) = {Mem(t), d(t), y(t), z(t), b(t)}
func (c *CacheEngine) GetCurrentState(requestedFile int) *State {
	F := c.Cfg.TotalFileTypes
	memRatio := 0.0
	if c.Cfg.CacheCapacity > 0 {
		memRatio = (c.Cfg.CacheCapacity - c.UsedCapacity) / c.Cfg.CacheCapacity
	}
	if memRatio < 0.0 {
		memRatio = 0.0
	}

	d := c.GetPopularityVector()
	y := make([]float64, F)
	z := make([]float64, F)
	b := make([]int, F)

	for i := 0; i < F && i < len(c.Files); i++ {
		y[i] = c.Files[i].Utility(c.CurrentTime, c.Cfg)
		z[i] = c.Files[i].Size
		if c.Cached[i] {
			b[i] = 1
		} else {
			b[i] = 0
		}
	}

	return &State{
		Mem:           memRatio,
		D:             d,
		Y:             y,
		Z:             z,
		B:             b,
		RequestedFile: requestedFile,
	}
}
