package config

import (
	"math"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.TotalFileTypes != 50 {
		t.Errorf("expected TotalFileTypes 50, got %d", cfg.TotalFileTypes)
	}
	if cfg.CacheCapacity != 10000.0 {
		t.Errorf("expected CacheCapacity 10000.0, got %f", cfg.CacheCapacity)
	}
	if cfg.SlidingWindowN != 100 {
		t.Errorf("expected SlidingWindowN 100, got %d", cfg.SlidingWindowN)
	}
	if cfg.UTMax != 1.5 || cfg.UTMin != 0.1 {
		t.Errorf("expected UTMax=1.5, UTMin=0.1, got %f, %f", cfg.UTMax, cfg.UTMin)
	}
	expectedCurve := (1.5 - 0.1) / (math.E - 1.0)
	if math.Abs(cfg.Curve-expectedCurve) > 1e-9 {
		t.Errorf("expected Curve %f, got %f", expectedCurve, cfg.Curve)
	}
	if cfg.DiscountGamma != 0.99 {
		t.Errorf("expected DiscountGamma 0.99, got %f", cfg.DiscountGamma)
	}
	if cfg.LambdaSource != 0.2 || cfg.LambdaTarget != 0.3 {
		t.Errorf("expected LambdaSource=0.2, LambdaTarget=0.3, got %f, %f", cfg.LambdaSource, cfg.LambdaTarget)
	}
	if cfg.BatchSize != 64 {
		t.Errorf("expected BatchSize 64, got %d", cfg.BatchSize)
	}
	if cfg.ReplayCapacity != 5000 || cfg.TargetBufferCap != 10000 {
		t.Errorf("expected ReplayCap=5000, TargetBufferCap=10000, got %d, %d", cfg.ReplayCapacity, cfg.TargetBufferCap)
	}
}

func TestDomainConfigs(t *testing.T) {
	src := SourceDomainConfig()
	if src.LambdaSource != 0.2 {
		t.Errorf("expected source lambda 0.2, got %f", src.LambdaSource)
	}
	tgt := TargetDomainConfig()
	if tgt.LambdaSource != 0.3 {
		t.Errorf("expected target lambda 0.3, got %f", tgt.LambdaSource)
	}
}

func TestConfigValidation(t *testing.T) {
	cfg := &Config{
		TotalFileTypes: -1,
		CacheCapacity:  -500,
		UTMax:          0.05,
		UTMin:          0.5,
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate returned error: %v", err)
	}
	if cfg.TotalFileTypes != 50 {
		t.Errorf("expected fallback TotalFileTypes=50, got %d", cfg.TotalFileTypes)
	}
	if cfg.CacheCapacity != 10000.0 {
		t.Errorf("expected fallback CacheCapacity=10000.0, got %f", cfg.CacheCapacity)
	}
	if cfg.UTMax != 1.5 || cfg.UTMin != 0.1 {
		t.Errorf("expected corrected UTMax=1.5, UTMin=0.1, got %f, %f", cfg.UTMax, cfg.UTMin)
	}
}
