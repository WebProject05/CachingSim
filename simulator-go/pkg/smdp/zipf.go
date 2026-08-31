package smdp

import (
	"math"
	"math/rand"
	"sync"

	"smdp-edge-caching-framework/pkg/config"
)

// Generator handles stochastic request generation for SMDP:
//   - Poisson inter-arrival intervals tau (Section III-B-1)
//   - Zipf file popularity sampling (Eq. 8)
type Generator struct {
	mu        sync.Mutex
	cfg       *config.Config
	rng       *rand.Rand
	zipfProbs []float64
	cdf       []float64
}

// NewGenerator initializes a new SMDP Generator with a random seed.
func NewGenerator(cfg *config.Config, seed int64) *Generator {
	rng := rand.New(rand.NewSource(seed))
	g := &Generator{
		cfg: cfg,
		rng: rng,
	}
	g.ComputeZipfDistribution(cfg.ZipfEta)
	return g
}

// ComputeZipfDistribution calculates p_f = 1 / (sigma * f^eta) (Eq. 8).
func (g *Generator) ComputeZipfDistribution(eta float64) {
	g.mu.Lock()
	defer g.mu.Unlock()

	F := g.cfg.TotalFileTypes
	if F <= 0 {
		g.zipfProbs = nil
		g.cdf = nil
		return
	}
	g.cfg.ZipfEta = eta
	g.zipfProbs = make([]float64, F)
	g.cdf = make([]float64, F)

	var sigma float64
	for f := 1; f <= F; f++ {
		sigma += math.Pow(float64(f), -eta)
	}

	if sigma <= 0 {
		sigma = 1.0
	}

	var cumulative float64
	invSigma := 1.0 / sigma
	for f := 1; f <= F; f++ {
		p := math.Pow(float64(f), -eta) * invSigma
		g.zipfProbs[f-1] = p
		cumulative += p
		g.cdf[f-1] = cumulative
	}
	if F > 0 {
		g.cdf[F-1] = 1.0
	}
}

// SampleRequestedFile selects a file type f_r in [0, F-1] sampled from the Zipf distribution (Eq. 8).
func (g *Generator) SampleRequestedFile() int {
	g.mu.Lock()
	defer g.mu.Unlock()

	n := len(g.cdf)
	if n == 0 {
		return -1
	}
	target := g.rng.Float64()
	lo, hi := 0, n-1
	for lo < hi {
		mid := int(uint(lo+hi) >> 1)
		if g.cdf[mid] < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo
}

// GetProbs returns a copy of the current Zipf probability vector.
func (g *Generator) GetProbs() []float64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	probs := make([]float64, len(g.zipfProbs))
	copy(probs, g.zipfProbs)
	return probs
}

// Reseed resets the internal PRNG source.
func (g *Generator) Reseed(seed int64) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.rng = rand.New(rand.NewSource(seed))
}
