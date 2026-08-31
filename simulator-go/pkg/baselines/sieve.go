package baselines

import (
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

// SIEVECache implements the SIEVE cache eviction algorithm.
// SIEVE maintains an eviction hand pointing to cache entries and uses a visited bit
// to give accessed entries a second chance, evicting the first unvisited item found.
type SIEVECache struct {
	cfg            *config.Config
	files          []core.FileMetadata
	capacity       float64
	usedCapacity   float64
	inCache        []bool
	visited        []bool
	prev           []int
	next           []int
	head           int
	tail           int
	hand           int
	cachedCount    int
	requests       int
	hits           int
	evictions      int
	rejected       int
	insertions     int
	requestedBytes float64
	hitBytes       float64
	missBytes      float64
}

// NewSIEVECache constructs a new SIEVECache.
func NewSIEVECache(cfg *config.Config, files []core.FileMetadata) *SIEVECache {
	n := len(files)
	prev := make([]int, n)
	next := make([]int, n)
	for i := 0; i < n; i++ {
		prev[i] = -1
		next[i] = -1
	}
	return &SIEVECache{
		cfg:          cfg,
		files:        files,
		capacity:     cfg.CacheCapacity,
		usedCapacity: 0.0,
		inCache:      make([]bool, n),
		visited:      make([]bool, n),
		prev:         prev,
		next:         next,
		head:         -1,
		tail:         -1,
		hand:         -1,
	}
}

// Access simulates a request for fileID. Returns true on cache hit, false on miss.
func (c *SIEVECache) Access(fileID int) bool {
	c.requests++
	if fileID < 0 || fileID >= len(c.files) {
		c.rejected++
		return false
	}
	fileSize := c.files[fileID].Size
	if fileSize > 0 {
		c.requestedBytes += fileSize
	}

	if c.inCache[fileID] {
		c.visited[fileID] = true
		c.hits++
		if fileSize > 0 {
			c.hitBytes += fileSize
		}
		return true
	}

	if fileSize > 0 {
		c.missBytes += fileSize
	}

	if fileSize < 0 || fileSize > c.capacity {
		c.rejected++
		return false
	}

	for c.capacity-c.usedCapacity < fileSize && c.cachedCount > 0 {
		c.evictOne()
	}

	if c.capacity-c.usedCapacity >= fileSize {
		c.pushBack(fileID)
		if c.hand < 0 {
			c.hand = fileID
		}
		c.usedCapacity += fileSize
		c.insertions++
	}

	return false
}

func (c *SIEVECache) pushBack(id int) {
	c.next[id] = -1
	c.prev[id] = c.tail
	if c.tail >= 0 {
		c.next[c.tail] = id
	} else {
		c.head = id
	}
	c.tail = id
	c.inCache[id] = true
	c.visited[id] = false
	c.cachedCount++
}

func (c *SIEVECache) evictOne() {
	if c.cachedCount == 0 {
		return
	}
	if c.hand < 0 {
		c.hand = c.head
	}

	for {
		id := c.hand
		next := c.next[id]
		if next < 0 {
			next = c.head
		}
		c.hand = next

		if c.visited[id] {
			c.visited[id] = false
			continue
		}

		c.remove(id)
		c.usedCapacity -= c.files[id].Size
		if c.usedCapacity < 0 {
			c.usedCapacity = 0
		}
		c.evictions++
		if c.cachedCount == 0 {
			c.hand = -1
		}
		return
	}
}

func (c *SIEVECache) remove(id int) {
	if c.prev[id] >= 0 {
		c.next[c.prev[id]] = c.next[id]
	} else {
		c.head = c.next[id]
	}
	if c.next[id] >= 0 {
		c.prev[c.next[id]] = c.prev[id]
	} else {
		c.tail = c.prev[id]
	}
	c.prev[id] = -1
	c.next[id] = -1
	c.inCache[id] = false
	c.visited[id] = false
	c.cachedCount--
}

// Stats returns the performance metrics collected by the SIEVE cache.
func (c *SIEVECache) Stats() CacheStats {
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
