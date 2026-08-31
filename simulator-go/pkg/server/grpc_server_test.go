package server

import (
	"context"
	"testing"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/pb"
)

func TestEnvServerLifecycle(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TotalFileTypes = 10
	cfg.CacheCapacity = 2000.0

	srv := NewEnvServer(cfg, 42)
	ctx := context.Background()

	// 1. Test Reset
	resetResp, err := srv.Reset(ctx, &pb.ResetRequest{
		Seed:   123,
		Lambda: 0.3,
		Eta:    0.8,
	})
	if err != nil {
		t.Fatalf("Reset returned error: %v", err)
	}
	if resetResp == nil {
		t.Fatal("Reset returned nil response")
	}
	if len(resetResp.D) != cfg.TotalFileTypes || len(resetResp.Y) != cfg.TotalFileTypes || len(resetResp.B) != cfg.TotalFileTypes {
		t.Fatalf("state vector dimensions mismatch: D=%d, Y=%d, B=%d", len(resetResp.D), len(resetResp.Y), len(resetResp.B))
	}
	if resetResp.Mem < 0.0 || resetResp.Mem > 1.0 {
		t.Fatalf("invalid Mem proportion: %f", resetResp.Mem)
	}

	// 2. Test Step with Action = 1 (Cache item)
	stepResp, err := srv.Step(ctx, &pb.StepRequest{Action: 1})
	if err != nil {
		t.Fatalf("Step returned error: %v", err)
	}
	if stepResp == nil {
		t.Fatal("Step returned nil response")
	}
	if stepResp.Tau <= 0.0 {
		t.Errorf("expected positive inter-arrival tau, got %f", stepResp.Tau)
	}
	if stepResp.NextState == nil {
		t.Fatal("Step returned nil NextState")
	}

	// 3. Test BatchStep
	batchResp, err := srv.BatchStep(ctx, &pb.BatchStepRequest{
		Actions: []int32{1, 0, 1, 1, 0},
	})
	if err != nil {
		t.Fatalf("BatchStep returned error: %v", err)
	}
	if len(batchResp.StepResponses) != 5 {
		t.Fatalf("expected 5 step responses, got %d", len(batchResp.StepResponses))
	}

	// 4. Test MDPStep
	mdpResp := srv.MDPStep(1, 0.4)
	if mdpResp == nil {
		t.Fatal("MDPStep returned nil response")
	}
	if mdpResp.Tau != 0.4 {
		t.Errorf("expected fixed time quota 0.4, got %f", mdpResp.Tau)
	}
}
