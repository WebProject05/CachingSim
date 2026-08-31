package smdp

import (
	"math"
)

// NextPoissonInterval returns continuous transition time tau = -ln(U) / lambda (Section III-B-1).
// If lambda <= 0, returns a default small interval.
func (g *Generator) NextPoissonInterval(lambda float64) float64 {
	if lambda <= 0 {
		return 1.0
	}
	g.mu.Lock()
	u := g.rng.Float64()
	for u <= 0.0 {
		u = g.rng.Float64()
	}
	g.mu.Unlock()
	return -math.Log(u) / lambda
}

// ExpectedPoissonInterval returns the theoretical mean inter-arrival time: tau_bar = 1 / lambda.
func ExpectedPoissonInterval(lambda float64) float64 {
	if lambda <= 0 {
		return 0.0
	}
	return 1.0 / lambda
}
