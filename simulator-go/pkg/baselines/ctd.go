package baselines

import (
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
	"smdp-edge-caching-framework/pkg/smdp"
)

// CTDCache implements the Caching Transient Data baseline (Zhu et al., IEEE IoT-J 2018 [17])
// as described in Section VII-C-1 and Section VII-E of the research paper.
//
// CTD characteristics:
//   - Formulates caching as a discrete-time MDP.
//   - Considers file popularity and freshness/lifetime, but ignores importance and variable sizes.
//   - Reward function focuses solely on the current requested item:
//     r_CTD = (1 - h^f(t)) on hit, and -1.0 (transmission delay penalty) on miss.
//   - Eviction policy prioritizes evicting expired or oldest/least-fresh items when full.
type CTDCache struct {
	cfg            *config.Config
	files          []core.FileMetadata
	capacity       float64
	usedCapacity   float64
	inCache        []bool
	cachedCount    int
	requests       int
	hits           int
	evictions      int
	rejected       int
	insertions     int
	requestedBytes float64
	hitBytes       float64
	missBytes      float64
	totalReward    float64
	totalUtility   float64
	currentTime    float64
}

// NewCTDCache constructs a new CTD baseline cache instance.
func NewCTDCache(cfg *config.Config, files []core.FileMetadata) *CTDCache {
	n := len(files)
	return &CTDCache{
		cfg:          cfg,
		files:        core.CloneFiles(files),
		capacity:     cfg.CacheCapacity,
		usedCapacity: 0.0,
		inCache:      make([]bool, n),
	}
}

// Access simulates a request for fileID under CTD policy.
func (c *CTDCache) Access(fileID int) bool {
	return c.AccessAtTime(fileID, c.currentTime)
}

// AccessAtTime processes a request at a specific simulation timestamp.
func (c *CTDCache) AccessAtTime(fileID int, currentTime float64) bool {
	c.requests++
	c.currentTime = currentTime

	if fileID < 0 || fileID >= len(c.files) {
		c.rejected++
		c.totalReward += -1.0
		return false
	}

	fileSize := c.files[fileID].Size
	if fileSize > 0 {
		c.requestedBytes += fileSize
	}

	// Update file generation time upon request (fetched fresh from data center)
	c.files[fileID].GenTime = currentTime

	// Check if already in cache and not expired
	if c.inCache[fileID] {
		freshness := c.files[fileID].Freshness(currentTime)
		if freshness < 1.0 {
			// Cache Hit
			c.hits++
			if fileSize > 0 {
				c.hitBytes += fileSize
			}
			reward := smdp.ComputeCTDReward(true, freshness)
			c.totalReward += reward
			c.totalUtility += c.files[fileID].Utility(currentTime, c.cfg)
			return true
		}
		// Stale/expired content treated as a miss in CTD
		c.remove(fileID)
	}

	// Cache Miss
	if fileSize > 0 {
		c.missBytes += fileSize
	}
	c.totalReward += smdp.ComputeCTDReward(false, 0.0)

	if fileSize > c.capacity {
		c.rejected++
		return false
	}

	// Evict least fresh (highest freshness h^f(t)) items until space is available
	for (c.capacity-c.usedCapacity) < fileSize && c.cachedCount > 0 {
		worstID := -1
		worstFreshness := -1.0

		for id := 0; id < len(c.inCache); id++ {
			if !c.inCache[id] {
				continue
			}
			h := c.files[id].Freshness(currentTime)
			if h > worstFreshness {
				worstFreshness = h
				worstID = id
			}
		}

		if worstID == -1 {
			break
		}
		c.remove(worstID)
		c.evictions++
	}

	if (c.capacity - c.usedCapacity) >= fileSize {
		c.inCache[fileID] = true
		c.usedCapacity += fileSize
		c.cachedCount++
		c.insertions++
	}

	return false
}

func (c *CTDCache) remove(id int) {
	if id >= 0 && id < len(c.inCache) && c.inCache[id] {
		c.inCache[id] = false
		c.usedCapacity -= c.files[id].Size
		if c.usedCapacity < 0 {
			c.usedCapacity = 0
		}
		c.cachedCount--
	}
}

// ComputeCTDReward calculates the instant reward for a given file and hit status.
func (c *CTDCache) ComputeCTDReward(fileID int, currentTime float64, isHit bool) float64 {
	if !isHit || fileID < 0 || fileID >= len(c.files) {
		return -1.0
	}
	freshness := c.files[fileID].Freshness(currentTime)
	return smdp.ComputeCTDReward(true, freshness)
}

// Stats returns standard cache statistics for the CTD baseline.
func (c *CTDCache) Stats() CacheStats {
	return CacheStats{
		Requests:         c.requests,
		Hits:             c.hits,
		Misses:           c.requests - c.hits,
		Evictions:        c.evictions,
		RejectedRequests: c.rejected,
		Insertions:       c.insertions,
		CachedFiles:      c.cachedCount,
		Capacity:         c.capacity,
		UsedCapacity:     c.usedCapacity,
		RequestedBytes:   c.requestedBytes,
		HitBytes:         c.hitBytes,
		MissBytes:        c.missBytes,
	}
}

// TotalReward returns the accumulated reward under CTD's reward formulation.
func (c *CTDCache) TotalReward() float64 {
	return c.totalReward
}

// TotalUtility returns the accumulated utility of all served requests.
func (c *CTDCache) TotalUtility() float64 {
	return c.totalUtility
}