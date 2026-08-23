package baselines

// CacheStats contains the measurements collected by a baseline run.
// Capacity and UsedCapacity are measured in MiB.
type CacheStats struct {
	Requests         int
	Hits             int
	Misses           int
	Evictions        int
	RejectedRequests int
	Insertions       int
	CachedFiles      int
	Capacity         float64
	UsedCapacity     float64
	RequestedBytes   float64
	HitBytes         float64
	MissBytes        float64
}

func (s CacheStats) HitRate() float64 {
	if s.Requests == 0 {
		return 0
	}
	return float64(s.Hits) / float64(s.Requests)
}

func (s CacheStats) MissRate() float64 {
	if s.Requests == 0 {
		return 0
	}
	return float64(s.Misses) / float64(s.Requests)
}

func (s CacheStats) RejectionRate() float64 {
	if s.Requests == 0 {
		return 0
	}
	return float64(s.RejectedRequests) / float64(s.Requests)
}

func (s CacheStats) EvictionRate() float64 {
	if s.Requests == 0 {
		return 0
	}
	return float64(s.Evictions) / float64(s.Requests)
}

func (s CacheStats) ByteHitRate() float64 {
	if s.RequestedBytes <= 0 {
		return 0
	}
	return s.HitBytes / s.RequestedBytes
}

func (s CacheStats) AverageRequestBytes() float64 {
	if s.Requests == 0 {
		return 0
	}
	return s.RequestedBytes / float64(s.Requests)
}

func (s CacheStats) Utilization() float64 {
	if s.Capacity <= 0 {
		return 0
	}
	return s.UsedCapacity / s.Capacity
}
