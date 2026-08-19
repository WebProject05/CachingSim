package baselines

import (
	"container/list"
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

type LRUCache struct {
	cfg          *config.Config
	files        []core.FileMetadata
	capacity     float64
	usedCapacity float64
	elements     map[int]*list.Element
	evictionList *list.List
	requests     int
	hits         int
	evictions    int
	rejected     int
}

func NewLRUCache(cfg *config.Config, files []core.FileMetadata) *LRUCache {
	return &LRUCache{
		cfg:          cfg,
		files:        files,
		capacity:     cfg.CacheCapacity,
		usedCapacity: 0.0,
		elements:     make(map[int]*list.Element),
		evictionList: list.New(),
	}
}

func (c *LRUCache) Access(fileID int) bool {
	c.requests++
	if fileID < 0 || fileID >= len(c.files) {
		c.rejected++
		return false
	}

	if elem, ok := c.elements[fileID]; ok {
		c.evictionList.MoveToFront(elem)
		c.hits++
		return true // Cache Hit
	}

	// Cache Miss -> Insert with LRU eviction
	fileSize := c.files[fileID].Size
	if fileSize < 0 || fileSize > c.capacity {
		c.rejected++
		return false
	}
	for (c.capacity-c.usedCapacity) < fileSize && c.evictionList.Len() > 0 {
		backElem := c.evictionList.Back()
		if backElem != nil {
			evictedID := backElem.Value.(int)
			c.evictionList.Remove(backElem)
			delete(c.elements, evictedID)
			c.usedCapacity -= c.files[evictedID].Size
			c.evictions++
		}
	}

	if (c.capacity - c.usedCapacity) >= fileSize {
		elem := c.evictionList.PushFront(fileID)
		c.elements[fileID] = elem
		c.usedCapacity += fileSize
	}

	return false
}

func (c *LRUCache) Stats() CacheStats {
	return CacheStats{
		Requests:         c.requests,
		Hits:             c.hits,
		Misses:           c.requests - c.hits,
		Evictions:        c.evictions,
		RejectedRequests: c.rejected,
		CachedFiles:      c.evictionList.Len(),
		Capacity:         c.capacity,
		UsedCapacity:     c.usedCapacity,
	}
}
