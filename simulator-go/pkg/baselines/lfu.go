package baselines

import (
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
	index   []int
}

func (h lfuHeap) less(first int, second int) bool {
	if h.entries[first].frequency != h.entries[second].frequency {
		return h.entries[first].frequency < h.entries[second].frequency
	}
	if h.entries[first].lastUsed != h.entries[second].lastUsed {
		return h.entries[first].lastUsed < h.entries[second].lastUsed
	}
	return h.entries[first].fileID < h.entries[second].fileID
}

func (h *lfuHeap) swap(first int, second int) {
	h.entries[first], h.entries[second] = h.entries[second], h.entries[first]
	h.entries[first].index = first
	h.entries[second].index = second
	h.index[h.entries[first].fileID] = first
	h.index[h.entries[second].fileID] = second
}

func (h *lfuHeap) up(i int) {
	for {
		parent := (i - 1) / 2
		if i == 0 || !h.less(i, parent) {
			break
		}
		h.swap(i, parent)
		i = parent
	}
}

func (h *lfuHeap) down(i int) {
	n := len(h.entries)
	for {
		left := 2*i + 1
		if left >= n {
			break
		}
		smallest := left
		if right := left + 1; right < n && h.less(right, left) {
			smallest = right
		}
		if !h.less(smallest, i) {
			break
		}
		h.swap(i, smallest)
		i = smallest
	}
}

func (h *lfuHeap) push(entry lfuEntry) {
	entry.index = len(h.entries)
	h.entries = append(h.entries, entry)
	h.index[entry.fileID] = entry.index
	h.up(entry.index)
}

func (h *lfuHeap) pop() lfuEntry {
	n := len(h.entries) - 1
	h.swap(0, n)
	last := h.entries[n]
	last.index = -1
	h.index[last.fileID] = -1
	h.entries = h.entries[:n]
	if n > 0 {
		h.down(0)
	}
	return last
}

func (h *lfuHeap) fix(i int) {
	if i > 0 && h.less(i, (i-1)/2) {
		h.up(i)
		return
	}
	h.down(i)
}

func (h *lfuHeap) Len() int { return len(h.entries) }

type LFUCache struct {
	cfg            *config.Config
	files          []core.FileMetadata
	capacity       float64
	usedCapacity   float64
	frequencies    []int
	cached         []bool
	lastUsed       []uint64
	evictionHeap   lfuHeap
	accessNumber   uint64
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

func NewLFUCache(cfg *config.Config, files []core.FileMetadata) *LFUCache {
	n := len(files)
	index := make([]int, n)
	for i := range index {
		index[i] = -1
	}
	return &LFUCache{
		cfg:          cfg,
		files:        files,
		capacity:     cfg.CacheCapacity,
		usedCapacity: 0.0,
		frequencies:  make([]int, n),
		cached:       make([]bool, n),
		lastUsed:     make([]uint64, n),
		evictionHeap: lfuHeap{entries: make([]lfuEntry, 0, n), index: index},
	}
}

func (c *LFUCache) Access(fileID int) bool {
	c.requests++
	if fileID < 0 || fileID >= len(c.files) {
		c.rejected++
		return false
	}

	fileSize := c.files[fileID].Size
	if fileSize > 0 {
		c.requestedBytes += fileSize
	}
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
		c.evictionHeap.fix(entryIndex)
		c.hits++
		if fileSize > 0 {
			c.hitBytes += fileSize
		}
		return true
	}
	if fileSize > 0 {
		c.missBytes += fileSize
	}
	for (c.capacity-c.usedCapacity) < fileSize && c.cachedCount > 0 {
		if c.evictionHeap.Len() == 0 {
			break
		}
		entry := c.evictionHeap.pop()
		lowestID := entry.fileID

		c.cached[lowestID] = false
		c.frequencies[lowestID] = 0
		c.lastUsed[lowestID] = 0
		c.cachedCount--
		c.usedCapacity -= c.files[lowestID].Size
		c.evictions++
	}

	if (c.capacity - c.usedCapacity) >= fileSize {
		c.cached[fileID] = true
		c.lastUsed[fileID] = c.accessNumber
		c.evictionHeap.push(lfuEntry{fileID: fileID, frequency: c.frequencies[fileID], lastUsed: c.accessNumber})
		c.usedCapacity += fileSize
		c.cachedCount++
		c.insertions++
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
		Insertions:       c.insertions,
		CachedFiles:      c.cachedCount,
		Capacity:         c.capacity,
		UsedCapacity:     c.usedCapacity,
		RequestedBytes:   c.requestedBytes,
		HitBytes:         c.hitBytes,
		MissBytes:        c.missBytes,
	}
}
