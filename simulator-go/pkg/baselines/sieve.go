package baselines

import (
	"container/list"
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

type sieveEntry struct {
	fileID  int
	visited bool
}

// SIEVECache uses a circular hand and one reference bit per cached file.
type SIEVECache struct {
	cfg          *config.Config
	files        []core.FileMetadata
	capacity     float64
	usedCapacity float64
	elements     map[int]*list.Element
	entries      *list.List
	hand         *list.Element
	requests     int
	hits         int
	evictions    int
	rejected     int
}

func NewSIEVECache(cfg *config.Config, files []core.FileMetadata) *SIEVECache {
	return &SIEVECache{
		cfg:      cfg,
		files:    files,
		capacity: cfg.CacheCapacity,
		elements: make(map[int]*list.Element),
		entries:  list.New(),
	}
}

func (c *SIEVECache) Access(fileID int) bool {
	c.requests++
	if fileID < 0 || fileID >= len(c.files) {
		c.rejected++
		return false
	}

	if elem, ok := c.elements[fileID]; ok {
		elem.Value.(*sieveEntry).visited = true
		c.hits++
		return true
	}

	fileSize := c.files[fileID].Size
	if fileSize < 0 || fileSize > c.capacity {
		c.rejected++
		return false
	}

	for c.capacity-c.usedCapacity < fileSize && c.entries.Len() > 0 {
		c.evictOne()
	}

	if c.capacity-c.usedCapacity >= fileSize {
		elem := c.entries.PushBack(&sieveEntry{fileID: fileID})
		c.elements[fileID] = elem
		if c.hand == nil {
			c.hand = elem
		}
		c.usedCapacity += fileSize
	}

	return false
}

func (c *SIEVECache) evictOne() {
	if c.entries.Len() == 0 {
		return
	}
	if c.hand == nil {
		c.hand = c.entries.Front()
	}

	for {
		elem := c.hand
		entry := elem.Value.(*sieveEntry)
		next := elem.Next()
		if next == nil {
			next = c.entries.Front()
		}
		c.hand = next

		if entry.visited {
			entry.visited = false
			continue
		}

		delete(c.elements, entry.fileID)
		c.entries.Remove(elem)
		c.usedCapacity -= c.files[entry.fileID].Size
		if c.usedCapacity < 0 {
			c.usedCapacity = 0
		}
		c.evictions++
		if c.entries.Len() == 0 {
			c.hand = nil
		}
		return
	}
}

func (c *SIEVECache) Stats() CacheStats {
	return CacheStats{
		Requests:         c.requests,
		Hits:             c.hits,
		Misses:           c.requests - c.hits,
		Evictions:        c.evictions,
		RejectedRequests: c.rejected,
		CachedFiles:      c.entries.Len(),
		Capacity:         c.capacity,
		UsedCapacity:     c.usedCapacity,
	}
}
