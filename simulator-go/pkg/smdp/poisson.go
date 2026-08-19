package smdp

import (
	"math"
)

// NextPoissonInterval returns continuous transition time tau = -ln(U) / lambda
func (g *Generator) NextPoissonInterval(lambda float64) float64 {
	if lambda <= 0 {
		return 0
	}
	u := g.rng.Float64()
	for u == 0.0 {
		u = g.rng.Float64()
	}
	return -math.Log(u) / lambda
}
