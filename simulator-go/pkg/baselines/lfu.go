package baselines

import (
	"container/heap"
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

type lfuEntry struct {
	fileID    int
	frequency int
	lastUsed  uint64
}

type lfuHeap []lfuEntry

func (h lfuHeap) Len() int { return len(h) }

func (h lfuHeap) Less(first int, second int) bool {
	if h[first].frequency != h[second].frequency {
		return h[first].frequency < h[second].frequency
	}
	if h[first].lastUsed != h[second].lastUsed {
		return h[first].lastUsed < h[second].lastUsed
	}
	return h[first].fileID < h[second].fileID
}

func (h lfuHeap) Swap(first int, second int) { h[first], h[second] = h[second], h[first] }

func (h *lfuHeap) Push(value interface{}) { *h = append(*h, value.(lfuEntry)) }

func (h *lfuHeap) Pop() interface{} {
	entries := *h
	last := entries[len(entries)-1]
	*h = entries[:len(entries)-1]
	return last
}

type LFUCache struct {
	cfg          *config.Config
	files        []core.FileMetadata
	capacity     float64
	usedCapacity float64
	frequencies  map[int]int
	cached       map[int]bool
	lastUsed     map[int]uint64
	evictionHeap lfuHeap
	accessNumber uint64
	requests     int
	hits         int
	evictions    int
	rejected     int
}

func NewLFUCache(cfg *config.Config, files []core.FileMetadata) *LFUCache {
	return &LFUCache{
		cfg:          cfg,
		files:        files,
		capacity:     cfg.CacheCapacity,
		usedCapacity: 0.0,
		frequencies:  make(map[int]int),
		cached:       make(map[int]bool),
		lastUsed:     make(map[int]uint64),
		evictionHeap: make(lfuHeap, 0),
	}
}

func (c *LFUCache) Access(fileID int) bool {
	c.requests++
	if fileID < 0 || fileID >= len(c.files) {
		c.rejected++
		return false
	}

	fileSize := c.files[fileID].Size
	if fileSize < 0 || fileSize > c.capacity {
		c.rejected++
		return false
	}

	c.accessNumber++
	c.frequencies[fileID]++
	if c.cached[fileID] {
		c.lastUsed[fileID] = c.accessNumber
		heap.Push(&c.evictionHeap, lfuEntry{fileID: fileID, frequency: c.frequencies[fileID], lastUsed: c.accessNumber})
		c.hits++
		return true
	}

	for (c.capacity-c.usedCapacity) < fileSize && len(c.cached) > 0 {
		lowestID := -1
		for c.evictionHeap.Len() > 0 {
			entry := heap.Pop(&c.evictionHeap).(lfuEntry)
			if frequency, ok := c.frequencies[entry.fileID]; ok &&
				c.lastUsed[entry.fileID] == entry.lastUsed && frequency == entry.frequency {
				lowestID = entry.fileID
				break
			}
		}

		if lowestID < 0 {
			break
		}

		delete(c.cached, lowestID)
		delete(c.frequencies, lowestID)
		delete(c.lastUsed, lowestID)
		c.usedCapacity -= c.files[lowestID].Size
		c.evictions++
	}

	if (c.capacity - c.usedCapacity) >= fileSize {
		c.cached[fileID] = true
		c.lastUsed[fileID] = c.accessNumber
		heap.Push(&c.evictionHeap, lfuEntry{fileID: fileID, frequency: c.frequencies[fileID], lastUsed: c.accessNumber})
		c.usedCapacity += fileSize
	}

	return false
}

func (c *LFUCache) Stats() CacheStats {
	return CacheStats{
		Requests:         c.requests,
		Hits:             c.hits,
		Misses:           c.requests - c.hits,
		Evictions:        c.evictions,
		RejectedRequests: c.rejected,
		CachedFiles:      len(c.cached),
		Capacity:         c.capacity,
		UsedCapacity:     c.usedCapacity,
	}
}
