package config

import (
	"math"
)

// MemoryUnit defines the unit used for all file sizes and cache capacity.
const MemoryUnit = "MiB"

// Config contains all simulation, environment, and reinforcement learning parameters
// defined in the research paper "Edge Caching Based on Deep Reinforcement Learning
// and Transfer Learning" (Niknia et al., 2024), specifically Table I and Table II.
type Config struct {
	// --- System & Environment Parameters (Table II) ---
	TotalFileTypes int     // F: Total number of file types (default: 50)
	CacheCapacity  float64 // M: Edge router cache capacity in MiB (default: 10000 MiB)
	SlidingWindowN int     // N: Sliding window size for popularity history d(t) (default: 100)

	// --- Utility Function Parameters (Eq. 1 & Table II) ---
	UTMax float64 // UT_max: Maximum utility coefficient (default: 1.5)
	UTMin float64 // UT_min: Minimum utility coefficient (default: 0.1)
	Curve float64 // Curve = (UT_max - UT_min) / (e - 1)

	// --- File Attribute Bounds (Table II) ---
	LifetimeMin   float64 // Min file lifetime w_l^f (default: 10.0)
	LifetimeMax   float64 // Max file lifetime w_l^f (default: 30.0)
	ImportanceMin float64 // Min file importance i_f (default: 0.1)
	ImportanceMax float64 // Max file importance i_f (default: 0.9)
	FileSizeMin   float64 // Min file size z_f in MiB (default: 100.0)
	FileSizeMax   float64 // Max file size z_f in MiB (default: 1000.0)

	// --- Arrival & Popularity Parameters (Table II) ---
	ZipfEta      float64 // eta: Zipf distribution skewness parameter in [0, 1] (default: 1.0)
	LambdaSource float64 // lambda_S: Request rate in source domain (default: 0.2)
	LambdaTarget float64 // lambda_T: Request rate in target domain (default: 0.3)

	// --- RL & SMDP Hyperparameters (Table II & Section V/VI) ---
	DiscountGamma     float64 // gamma: Discount factor (default: 0.99)
	LearningRate      float64 // psi: Learning rate for neural networks (default: 0.0005)
	TargetSyncSteps   int     // zeta: Target network update frequency (default: 100)
	BatchSize         int     // Mini-batch size for training (default: 64)
	ReplayCapacity    int     // |Lambda|: Replay buffer size in source domain (default: 5000)
	TargetBufferCap   int     // |Lambda'|: Replay buffer size in target domain (default: 10000)
	PrioritizedAlpha  float64 // alpha: Priority exponent for sampling in PER (default: 0.4)
	BetaInitial       float64 // beta: Initial importance sampling weight exponent (default: 0.6)
	BetaFinal         float64 // beta: Final importance sampling weight exponent (default: 1.0)
	PriorityEpsilon   float64 // varsigma: Small positive constant for PER (default: 1e-5)
	NStepHorizon      int     // n: Multi-step horizon for n-step return J_n (default: 3)
	LossWeightNStep   float64 // lambda_1: Weight for n-step TD loss (default: 1.0)
	LossWeightExpert  float64 // lambda_2: Weight for large margin supervised loss J_E (default: 1.0)
	LossWeightL2      float64 // lambda_3: Weight for L2 regularization J_L2 (default: 1e-5)
	ExpertMargin      float64 // Supervised margin penalty L(a_E, a) (default: 0.8)
}

// ComputeCurve calculates the Curve coefficient: (UT_max - UT_min) / (e - 1).
func ComputeCurve(utMax, utMin float64) float64 {
	return (utMax - utMin) / (math.E - 1.0)
}

// DefaultConfig returns the default configuration specified in the research paper (Table II).
func DefaultConfig() *Config {
	utMax := 1.5
	utMin := 0.1
	return &Config{
		TotalFileTypes:   50,
		CacheCapacity:    10000.0,
		SlidingWindowN:   100,
		UTMax:            utMax,
		UTMin:            utMin,
		Curve:            ComputeCurve(utMax, utMin),
		LifetimeMin:      10.0,
		LifetimeMax:      30.0,
		ImportanceMin:    0.1,
		ImportanceMax:    0.9,
		FileSizeMin:      100.0,
		FileSizeMax:      1000.0,
		ZipfEta:          1.0,
		LambdaSource:     0.2,
		LambdaTarget:     0.3,
		DiscountGamma:    0.99,
		LearningRate:     0.0005,
		TargetSyncSteps:  100,
		BatchSize:        64,
		ReplayCapacity:   5000,
		TargetBufferCap:  10000,
		PrioritizedAlpha: 0.4,
		BetaInitial:      0.6,
		BetaFinal:        1.0,
		PriorityEpsilon:  1e-5,
		NStepHorizon:     3,
		LossWeightNStep:  1.0,
		LossWeightExpert: 1.0,
		LossWeightL2:     1e-5,
		ExpertMargin:     0.8,
	}
}

// SourceDomainConfig returns a configuration tailored for source domain training (lambda = 0.2).
func SourceDomainConfig() *Config {
	cfg := DefaultConfig()
	cfg.LambdaSource = 0.2
	return cfg
}

// TargetDomainConfig returns a configuration tailored for target domain transfer learning (lambda = 0.3).
func TargetDomainConfig() *Config {
	cfg := DefaultConfig()
	cfg.LambdaSource = 0.3
	return cfg
}

// Validate ensures all configuration parameters are positive and well-formed.
func (c *Config) Validate() error {
	if c.TotalFileTypes <= 0 {
		c.TotalFileTypes = 50
	}
	if c.CacheCapacity <= 0 {
		c.CacheCapacity = 10000.0
	}
	if c.SlidingWindowN <= 0 {
		c.SlidingWindowN = 100
	}
	if c.UTMax <= c.UTMin {
		c.UTMax = 1.5
		c.UTMin = 0.1
	}
	c.Curve = ComputeCurve(c.UTMax, c.UTMin)
	if c.LifetimeMin <= 0 || c.LifetimeMax <= c.LifetimeMin {
		c.LifetimeMin = 10.0
		c.LifetimeMax = 30.0
	}
	if c.ImportanceMin <= 0 || c.ImportanceMax <= c.ImportanceMin {
		c.ImportanceMin = 0.1
		c.ImportanceMax = 0.9
	}
	if c.FileSizeMin <= 0 || c.FileSizeMax <= c.FileSizeMin {
		c.FileSizeMin = 100.0
		c.FileSizeMax = 1000.0
	}
	if c.DiscountGamma <= 0 || c.DiscountGamma > 1.0 {
		c.DiscountGamma = 0.99
	}
	if c.LambdaSource <= 0 {
		c.LambdaSource = 0.2
	}
	if c.LambdaTarget <= 0 {
		c.LambdaTarget = 0.3
	}
	return nil
}
