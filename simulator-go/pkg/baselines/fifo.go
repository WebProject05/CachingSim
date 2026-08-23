package baselines

import (
	"container/list"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

// FIFOCache evicts cached files in the order they were inserted.
type FIFOCache struct {
	cfg            *config.Config
	files          []core.FileMetadata
	capacity       float64
	usedCapacity   float64
	queue          *list.List
	entries        map[int]*list.Element
	requests       int
	hits           int
	evictions      int
	rejected       int
	insertions     int
	cachedCount    int
	requestedBytes float64
	hitBytes       float64
	missBytes      float64
}

func NewFIFOCache(cfg *config.Config, files []core.FileMetadata) *FIFOCache {
	return &FIFOCache{
		cfg:      cfg,
		files:    files,
		capacity: cfg.CacheCapacity,
		queue:    list.New(),
		entries:  make(map[int]*list.Element, len(files)),
	}
}

func (c *FIFOCache) Access(fileID int) bool {
	c.requests++
	if fileID < 0 || fileID >= len(c.files) {
		c.rejected++
		return false
	}

	fileSize := c.files[fileID].Size
	if fileSize > 0 {
		c.requestedBytes += fileSize
	}
	if _, ok := c.entries[fileID]; ok {
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

	for c.capacity-c.usedCapacity < fileSize && c.queue.Len() > 0 {
		oldest := c.queue.Front()
		oldestID := oldest.Value.(int)
		c.queue.Remove(oldest)
		delete(c.entries, oldestID)
		c.usedCapacity -= c.files[oldestID].Size
		c.cachedCount--
		c.evictions++
	}
	if c.capacity-c.usedCapacity < fileSize {
		return false
	}

	c.entries[fileID] = c.queue.PushBack(fileID)
	c.usedCapacity += fileSize
	c.cachedCount++
	c.insertions++
	return false
}

func (c *FIFOCache) Stats() CacheStats {
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
