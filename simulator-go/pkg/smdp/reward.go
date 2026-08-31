package smdp

import (
	"math"
	"smdp-edge-caching-framework/pkg/core"
)

// ComputeInstantReward calculates the SMDP instant reward r(t) (Eq. 2):
//
//	r(t) = W(t) - Mem(t) * 100
//
// Where W(t) is the total worth of cached files (Eq. 3).
func ComputeInstantReward(worth, memRatio float64) float64 {
	return worth - (memRatio * 100.0)
}

// ComputeWorth calculates total worth of cached files W(t) (Eq. 3):
//
//	W(t) = (b(t) . d(t)) (b(t) . y(t))^T = sum_{f=1}^F b_f(t) * d_f(t) * y_f(t)
func ComputeWorth(cached []bool, popularity []float64, utilities []float64) float64 {
	return core.CalculateCachedWorth(cached, popularity, utilities)
}

// ContinuousDiscount calculates gamma^tau for continuous-time SMDP Bellman updates (Section V-A).
func ContinuousDiscount(gamma, tau float64) float64 {
	if gamma <= 0.0 {
		return 0.0
	}
	if tau <= 0.0 {
		return 1.0
	}
	return math.Pow(gamma, tau)
}

// ComputeNStepDiscountedReturn calculates R_t^(n) and total duration tau^(n) (Eq. 7):
//
//	tau^(n) = tau_1 + tau_2 + ... + tau_n
//	R_t^(n) = gamma^tau(1) * r(1) + gamma^tau(2) * r(2) + ... + gamma^tau(n) * r(n)
func ComputeNStepDiscountedReturn(rewards []float64, taus []float64, gamma float64) (float64, float64) {
	n := len(rewards)
	if n == 0 || len(taus) != n {
		return 0.0, 0.0
	}

	var cumulativeTau float64
	var nStepReturn float64

	for i := 0; i < n; i++ {
		cumulativeTau += taus[i]
		discount := ContinuousDiscount(gamma, cumulativeTau)
		nStepReturn += discount * rewards[i]
	}

	return nStepReturn, cumulativeTau
}

// ComputeProposedPriority calculates the sampling priority p_i using proposed Eq. (6):
//
//	p_i = (avgR(t) - r_i) + |TDE_i| + varsigma
//
// Where avgR(t) is the running average reward up to time t, r_i is transition reward,
// TDE_i is Temporal Difference Error, and varsigma is a small positive constant.
func ComputeProposedPriority(avgReward, reward, tde, epsilon float64) float64 {
	p := (avgReward - reward) + math.Abs(tde) + epsilon
	if p <= epsilon {
		return epsilon
	}
	return p
}

// ComputeStandardPriority calculates standard DQfD priority using Eq. (5):
//
//	p_i = |TDE_i| + varsigma
func ComputeStandardPriority(tde, epsilon float64) float64 {
	p := math.Abs(tde) + epsilon
	if p <= epsilon {
		return epsilon
	}
	return p
}

// ComputeImportanceSamplingWeight calculates omega_i (Section VI-1):
//
//	omega_i = ( 1 / (|Lambda'| * P(i)) )^beta
func ComputeImportanceSamplingWeight(prob float64, bufferSize int, beta float64) float64 {
	if prob <= 0.0 || bufferSize <= 0 {
		return 1.0
	}
	return math.Pow(1.0/(float64(bufferSize)*prob), beta)
}

// ComputeCTDReward calculates the Caching Transient Data (CTD) baseline reward (Zhu et al. [17]):
//   - On Cache Hit: 1.0 - freshness (reward for data freshness)
//   - On Cache Miss: -1.0 (penalty for transmission delay)
func ComputeCTDReward(isHit bool, freshness float64) float64 {
	if !isHit {
		return -1.0
	}
	if freshness < 0.0 {
		freshness = 0.0
	} else if freshness > 1.0 {
		freshness = 1.0
	}
	return 1.0 - freshness
}

// RunningAverageReward tracks cumulative average reward: avgR(t) = sum_{i=0}^t r_i / (t + 1).
type RunningAverageReward struct {
	count int
	sum   float64
}

// NewRunningAverageReward creates a new RunningAverageReward tracker.
func NewRunningAverageReward() *RunningAverageReward {
	return &RunningAverageReward{}
}

// Update records a new reward and returns the updated average.
func (r *RunningAverageReward) Update(reward float64) float64 {
	r.count++
	r.sum += reward
	return r.sum / float64(r.count)
}

// Value returns the current average reward.
func (r *RunningAverageReward) Value() float64 {
	if r.count == 0 {
		return 0.0
	}
	return r.sum / float64(r.count)
}

// Reset resets the running average tracker.
func (r *RunningAverageReward) Reset() {
	r.count = 0
	r.sum = 0.0
}
