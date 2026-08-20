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
	index     int
}

type lfuHeap struct {
	entries []lfuEntry
	index   map[int]int
}

func (h lfuHeap) Len() int { return len(h.entries) }

func (h lfuHeap) Less(first int, second int) bool {
	if h.entries[first].frequency != h.entries[second].frequency {
		return h.entries[first].frequency < h.entries[second].frequency
	}
	if h.entries[first].lastUsed != h.entries[second].lastUsed {
		return h.entries[first].lastUsed < h.entries[second].lastUsed
	}
	return h.entries[first].fileID < h.entries[second].fileID
}

func (h lfuHeap) Swap(first int, second int) {
	h.entries[first], h.entries[second] = h.entries[second], h.entries[first]
	h.entries[first].index = first
	h.entries[second].index = second
	h.index[h.entries[first].fileID] = first
	h.index[h.entries[second].fileID] = second
}

func (h *lfuHeap) Push(value interface{}) {
	entry := value.(lfuEntry)
	entry.index = len(h.entries)
	h.entries = append(h.entries, entry)
	h.index[entry.fileID] = entry.index
}

func (h *lfuHeap) Pop() interface{} {
	entries := h.entries
	lastIndex := len(entries) - 1
	last := entries[lastIndex]
	last.index = -1
	delete(h.index, last.fileID)
	h.entries = entries[:len(entries)-1]
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
		evictionHeap: lfuHeap{entries: make([]lfuEntry, 0), index: make(map[int]int)},
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
		entryIndex := c.evictionHeap.index[fileID]
		c.evictionHeap.entries[entryIndex].frequency = c.frequencies[fileID]
		c.evictionHeap.entries[entryIndex].lastUsed = c.accessNumber
		heap.Fix(&c.evictionHeap, entryIndex)
		c.hits++
		return true
	}

	for (c.capacity-c.usedCapacity) < fileSize && len(c.cached) > 0 {
		if c.evictionHeap.Len() == 0 {
			break
		}
		entry := heap.Pop(&c.evictionHeap).(lfuEntry)
		lowestID := entry.fileID

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
