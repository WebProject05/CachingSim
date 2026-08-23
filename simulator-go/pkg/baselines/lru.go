package baselines

import (
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

type LRUCache struct {
	cfg            *config.Config
	files          []core.FileMetadata
	capacity       float64
	usedCapacity   float64
	inCache        []bool
	prev           []int
	next           []int
	head           int
	tail           int
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

func NewLRUCache(cfg *config.Config, files []core.FileMetadata) *LRUCache {
	n := len(files)
	prev := make([]int, n)
	next := make([]int, n)
	for i := 0; i < n; i++ {
		prev[i] = -1
		next[i] = -1
	}
	return &LRUCache{
		cfg:          cfg,
		files:        files,
		capacity:     cfg.CacheCapacity,
		usedCapacity: 0.0,
		inCache:      make([]bool, n),
		prev:         prev,
		next:         next,
		head:         -1,
		tail:         -1,
	}
}

func (c *LRUCache) Access(fileID int) bool {
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
		c.moveToFront(fileID)
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
	for (c.capacity-c.usedCapacity) < fileSize && c.tail >= 0 {
		evictedID := c.tail
		c.remove(evictedID)
		c.usedCapacity -= c.files[evictedID].Size
		c.evictions++
	}

	if (c.capacity - c.usedCapacity) >= fileSize {
		c.pushFront(fileID)
		c.usedCapacity += fileSize
		c.insertions++
	}

	return false
}

func (c *LRUCache) pushFront(id int) {
	c.prev[id] = -1
	c.next[id] = c.head
	if c.head >= 0 {
		c.prev[c.head] = id
	} else {
		c.tail = id
	}
	c.head = id
	c.inCache[id] = true
	c.cachedCount++
}

func (c *LRUCache) moveToFront(id int) {
	if c.head == id {
		return
	}
	c.unlink(id)
	c.prev[id] = -1
	c.next[id] = c.head
	c.prev[c.head] = id
	c.head = id
}

func (c *LRUCache) remove(id int) {
	c.unlink(id)
	c.prev[id] = -1
	c.next[id] = -1
	c.inCache[id] = false
	c.cachedCount--
}

func (c *LRUCache) unlink(id int) {
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
}

func (c *LRUCache) Stats() CacheStats {
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
