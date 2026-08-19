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
		return
	}
	eta := g.cfg.ZipfEta
	g.zipfProbs = make([]float64, F)

	var sigma float64
	for f := 1; f <= F; f++ {
		sigma += 1.0 / math.Pow(float64(f), eta)
	}

	for f := 1; f <= F; f++ {
		g.zipfProbs[f-1] = (1.0 / math.Pow(float64(f), eta)) / sigma
	}
}

// SampleRequestedFile selects a file type f_r sampled from the Zipf distribution
func (g *Generator) SampleRequestedFile() int {
	if len(g.zipfProbs) == 0 {
		return -1
	}
	target := g.rng.Float64()
	var cumulative float64
	for i, p := range g.zipfProbs {
		cumulative += p
		if target <= cumulative {
			return i
		}
	}
	return g.cfg.TotalFileTypes - 1
}
