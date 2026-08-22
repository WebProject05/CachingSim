package smdp

import (
	"math"
	"math/rand"
	"smdp-edge-caching-framework/pkg/config"
)

type Generator struct {
	cfg       *config.Config
	rng       *rand.Rand
	zipfProbs []float64
	cdf       []float64
}

func NewGenerator(cfg *config.Config, seed int64) *Generator {
	rng := rand.New(rand.NewSource(seed))
	g := &Generator{cfg: cfg, rng: rng}
	g.computeZipfDistribution()
	return g
}

// computeZipfDistribution calculates p_f = 1 / (sigma * f^eta) (Eq. 8)
func (g *Generator) computeZipfDistribution() {
	F := g.cfg.TotalFileTypes
	if F <= 0 {
		g.zipfProbs = nil
		g.cdf = nil
		return
	}
	eta := g.cfg.ZipfEta
	g.zipfProbs = make([]float64, F)
	g.cdf = make([]float64, F)

	var sigma float64
	for f := 1; f <= F; f++ {
		sigma += math.Pow(float64(f), -eta)
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

// SampleRequestedFile selects a file type f_r sampled from the Zipf distribution
func (g *Generator) SampleRequestedFile() int {
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
