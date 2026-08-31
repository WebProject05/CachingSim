package smdp

import (
	"math"
	"testing"

	"smdp-edge-caching-framework/pkg/config"
)

func TestPoissonIntervalGeneration(t *testing.T) {
	cfg := config.DefaultConfig()
	gen := NewGenerator(cfg, 42)

	lambda := 0.2
	trials := 10000
	var sum float64

	for i := 0; i < trials; i++ {
		tau := gen.NextPoissonInterval(lambda)
		if tau <= 0.0 {
			t.Fatalf("poisson interval must be positive, got %f", tau)
		}
		sum += tau
	}

	mean := sum / float64(trials)
	expectedMean := 1.0 / lambda // 5.0
	// Within 5% of theoretical mean with 10k samples
	if math.Abs(mean-expectedMean)/expectedMean > 0.05 {
		t.Errorf("empirical mean interval %f differed significantly from expected %f", mean, expectedMean)
	}
}

func TestZipfDistribution(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TotalFileTypes = 10
	cfg.ZipfEta = 1.0

	gen := NewGenerator(cfg, 42)
	probs := gen.GetProbs()

	if len(probs) != 10 {
		t.Fatalf("expected 10 probabilities, got %d", len(probs))
	}

	var sum float64
	for i, p := range probs {
		if p <= 0 {
			t.Errorf("probability for file %d must be positive, got %f", i, p)
		}
		if i > 0 && probs[i] >= probs[i-1] {
			t.Errorf("Zipf probability should be strictly decreasing with rank: p[%d]=%f >= p[%d]=%f", i, probs[i], i-1, probs[i-1])
		}
		sum += p
	}

	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("probabilities must sum to 1.0, got %f", sum)
	}
}

func TestZipfSamplingSkewness(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TotalFileTypes = 5
	cfg.ZipfEta = 1.5

	gen := NewGenerator(cfg, 42)
	counts := make([]int, 5)
	trials := 20000

	for i := 0; i < trials; i++ {
		f := gen.SampleRequestedFile()
		if f < 0 || f >= 5 {
			t.Fatalf("sampled file %d out of range [0, 4]", f)
		}
		counts[f]++
	}

	// File 0 should have the highest count
	for i := 1; i < 5; i++ {
		if counts[i] >= counts[i-1] {
			t.Errorf("empirical sampling not skewed: count[%d]=%d >= count[%d]=%d", i, counts[i], i-1, counts[i-1])
		}
	}
}

func TestInstantRewardCalculation(t *testing.T) {
	worth := 45.0
	memRatio := 0.2 // 20% free memory
	r := ComputeInstantReward(worth, memRatio)
	expectedR := 45.0 - (0.2 * 100.0) // 45.0 - 20.0 = 25.0
	if math.Abs(r-expectedR) > 1e-9 {
		t.Errorf("expected reward %f, got %f", expectedR, r)
	}
}

func TestNStepDiscountedReturn(t *testing.T) {
	gamma := 0.99
	rewards := []float64{10.0, 20.0, 30.0}
	taus := []float64{1.0, 2.0, 1.5}

	nStepReturn, cumTau := ComputeNStepDiscountedReturn(rewards, taus, gamma)

	expectedCumTau := 1.0 + 2.0 + 1.5 // 4.5
	if math.Abs(cumTau-expectedCumTau) > 1e-9 {
		t.Errorf("expected cumTau %f, got %f", expectedCumTau, cumTau)
	}

	expectedReturn := math.Pow(gamma, 1.0)*10.0 +
		math.Pow(gamma, 1.0+2.0)*20.0 +
		math.Pow(gamma, 1.0+2.0+1.5)*30.0

	if math.Abs(nStepReturn-expectedReturn) > 1e-6 {
		t.Errorf("expected nStepReturn %f, got %f", expectedReturn, nStepReturn)
	}
}

func TestPERPriorities(t *testing.T) {
	avgReward := 50.0
	reward := 30.0
	tde := 2.5
	epsilon := 1e-5

	// Proposed Priority Eq. (6): (avgR - r) + |TDE| + epsilon
	pProposed := ComputeProposedPriority(avgReward, reward, tde, epsilon)
	expectedProposed := (50.0 - 30.0) + 2.5 + 1e-5 // 22.50001
	if math.Abs(pProposed-expectedProposed) > 1e-6 {
		t.Errorf("expected proposed priority %f, got %f", expectedProposed, pProposed)
	}

	// Standard Priority Eq. (5): |TDE| + epsilon
	pStd := ComputeStandardPriority(tde, epsilon)
	expectedStd := 2.5 + 1e-5
	if math.Abs(pStd-expectedStd) > 1e-6 {
		t.Errorf("expected standard priority %f, got %f", expectedStd, pStd)
	}
}

func TestCTDReward(t *testing.T) {
	// Hit with freshness 0.2 -> 1 - 0.2 = 0.8
	rHit := ComputeCTDReward(true, 0.2)
	if math.Abs(rHit-0.8) > 1e-9 {
		t.Errorf("expected CTD hit reward 0.8, got %f", rHit)
	}

	// Miss -> -1.0
	rMiss := ComputeCTDReward(false, 0.2)
	if rMiss != -1.0 {
		t.Errorf("expected CTD miss penalty -1.0, got %f", rMiss)
	}
}

func TestRunningAverageReward(t *testing.T) {
	tracker := NewRunningAverageReward()
	tracker.Update(10.0)
	tracker.Update(20.0)
	tracker.Update(30.0)

	avg := tracker.Value()
	if math.Abs(avg-20.0) > 1e-9 {
		t.Errorf("expected running average 20.0, got %f", avg)
	}
}
