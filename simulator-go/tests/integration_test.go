package tests

import (
	"context"
	"testing"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/pb"
	"smdp-edge-caching-framework/pkg/server"
)

func TestEndToEndSMDPSimulation(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TotalFileTypes = 20
	cfg.CacheCapacity = 5000.0
	cfg.LambdaSource = 0.2
	cfg.ZipfEta = 1.0

	srv := server.NewEnvServer(cfg, 42)
	ctx := context.Background()

	// Reset environment
	stateResp, err := srv.Reset(ctx, &pb.ResetRequest{Seed: 42, Lambda: 0.2, Eta: 1.0})
	if err != nil {
		t.Fatalf("Reset failed: %v", err)
	}
	if stateResp == nil {
		t.Fatal("nil StateResponse on reset")
	}

	totalSteps := 500
	hitCount := 0
	totalReward := 0.0

	for step := 0; step < totalSteps; step++ {
		// Greedy heuristic policy: always cache (action = 1)
		resp, err := srv.Step(ctx, &pb.StepRequest{Action: 1})
		if err != nil {
			t.Fatalf("Step %d failed: %v", step, err)
		}
		if resp.IsHit {
			hitCount++
		}
		totalReward += resp.Reward

		// Validate state boundaries
		if resp.NextState.Mem < 0.0 || resp.NextState.Mem > 1.0 {
			t.Errorf("Step %d: invalid memory ratio: %f", step, resp.NextState.Mem)
		}
		if resp.Tau <= 0.0 {
			t.Errorf("Step %d: non-positive Poisson tau: %f", step, resp.Tau)
		}
	}

	hitRate := float64(hitCount) / float64(totalSteps) * 100.0
	if hitRate <= 0.0 {
		t.Errorf("expected positive hit rate, got %f%%", hitRate)
	}
	avgReward := totalReward / float64(totalSteps)
	if avgReward == 0.0 {
		t.Errorf("expected non-zero average reward")
	}
}

func TestDomainTransferAdaptation(t *testing.T) {
	cfg := config.DefaultConfig()
	srv := server.NewEnvServer(cfg, 42)
	ctx := context.Background()

	// Source domain run (lambda = 0.2)
	_, err := srv.Reset(ctx, &pb.ResetRequest{Seed: 42, Lambda: 0.2, Eta: 1.0})
	if err != nil {
		t.Fatalf("Source domain reset failed: %v", err)
	}

	var sourceTaus float64
	for i := 0; i < 200; i++ {
		resp, err := srv.Step(ctx, &pb.StepRequest{Action: 1})
		if err != nil {
			t.Fatalf("source step failed: %v", err)
		}
		sourceTaus += resp.Tau
	}
	avgSourceTau := sourceTaus / 200.0

	// Target domain run (lambda = 0.5 - higher request rate implies shorter tau)
	_, err = srv.Reset(ctx, &pb.ResetRequest{Seed: 42, Lambda: 0.5, Eta: 1.0})
	if err != nil {
		t.Fatalf("Target domain reset failed: %v", err)
	}

	var targetTaus float64
	for i := 0; i < 200; i++ {
		resp, err := srv.Step(ctx, &pb.StepRequest{Action: 1})
		if err != nil {
			t.Fatalf("target step failed: %v", err)
		}
		targetTaus += resp.Tau
	}
	avgTargetTau := targetTaus / 200.0

	if avgTargetTau >= avgSourceTau {
		t.Errorf("target domain mean tau (%f) should be smaller than source domain mean tau (%f)", avgTargetTau, avgSourceTau)
	}
}
