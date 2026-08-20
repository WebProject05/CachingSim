package baselines

// CacheStats contains the measurements collected by a baseline run.
// Capacity and UsedCapacity are measured in MiB.
type CacheStats struct {
	Requests         int
	Hits             int
	Misses           int
	Evictions        int
	RejectedRequests int
	CachedFiles      int
	Capacity         float64
	UsedCapacity     float64
}

func (s CacheStats) HitRate() float64 {
	if s.Requests == 0 {
		return 0
	}
	return float64(s.Hits) / float64(s.Requests)
}

func (s CacheStats) Utilization() float64 {
	if s.Capacity <= 0 {
		return 0
	}
	return s.UsedCapacity / s.Capacity
}
