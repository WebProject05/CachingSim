package server

import (
	"context"
	"math/rand"
	"sync"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
	"smdp-edge-caching-framework/pkg/pb"
	"smdp-edge-caching-framework/pkg/smdp"
)

// EnvServer implements the gRPC CacheEnvServiceServer defined in proto/cache_env.proto.
// It bridges the Go SMDP simulation engine with reinforcement learning / transfer learning agents.
type EnvServer struct {
	pb.UnimplementedCacheEnvServiceServer
	mu         sync.Mutex
	cfg        *config.Config
	cache      *core.CacheEngine
	generator  *smdp.Generator
	currentReq int
}

// NewEnvServer constructs and initializes an EnvServer instance with given config and random seed.
func NewEnvServer(cfg *config.Config, seed int64) *EnvServer {
	if seed == 0 {
		seed = 42
	}
	cfgCopy := *cfg
	rng := rand.New(rand.NewSource(seed))
	files := core.GenerateFiles(&cfgCopy, rng)
	cache := core.NewCacheEngine(&cfgCopy, files)
	generator := smdp.NewGenerator(&cfgCopy, seed)
	currentReq := generator.SampleRequestedFile()
	cache.RecordRequest(currentReq)

	return &EnvServer{
		cfg:        &cfgCopy,
		cache:      cache,
		generator:  generator,
		currentReq: currentReq,
	}
}

// Reset re-initializes the environment state, random seed, and optional domain parameters (lambda, eta).
func (s *EnvServer) Reset(ctx context.Context, req *pb.ResetRequest) (*pb.StateResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	seed := req.Seed
	if seed == 0 {
		seed = 42
	}

	if req.Lambda > 0 {
		s.cfg.LambdaSource = req.Lambda
	}
	if req.Eta > 0 {
		s.cfg.ZipfEta = req.Eta
	}

	rng := rand.New(rand.NewSource(seed))
	files := core.GenerateFiles(s.cfg, rng)
	s.cache = core.NewCacheEngine(s.cfg, files)
	s.generator = smdp.NewGenerator(s.cfg, seed)
	s.currentReq = s.generator.SampleRequestedFile()
	s.cache.RecordRequest(s.currentReq)

	state := s.cache.GetCurrentState(s.currentReq)
	return mapStateToProto(state, s.cache.CurrentTime), nil
}

// Step advances the environment by taking action a(t) in {0, 1} for current request f_r.
func (s *EnvServer) Step(ctx context.Context, req *pb.StepRequest) (*pb.StepResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	tau := s.generator.NextPoissonInterval(s.cfg.LambdaSource)
	return s.executeStepLocked(req.Action, tau), nil
}

// BatchStep processes multiple sequential actions atomically and returns a slice of StepResponses.
func (s *EnvServer) BatchStep(ctx context.Context, req *pb.BatchStepRequest) (*pb.BatchStepResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	responses := make([]*pb.StepResponse, len(req.Actions))
	for i, action := range req.Actions {
		tau := s.generator.NextPoissonInterval(s.cfg.LambdaSource)
		responses[i] = s.executeStepLocked(action, tau)
	}
	return &pb.BatchStepResponse{StepResponses: responses}, nil
}

// MDPStep executes a step with a fixed discrete time quota instead of continuous Poisson tau.
func (s *EnvServer) MDPStep(action int32, timeQuota float64) *pb.StepResponse {
	s.mu.Lock()
	defer s.mu.Unlock()

	if timeQuota <= 0 {
		timeQuota = 0.2
	}
	return s.executeStepLocked(action, timeQuota)
}

// executeStepLocked handles state transition, caching, reward, and clock advance.
func (s *EnvServer) executeStepLocked(action int32, tau float64) *pb.StepResponse {
	reqFile := s.currentReq
	isHit := s.cache.IsCached(reqFile)

	var utilityGained float64
	if isHit && reqFile >= 0 && reqFile < len(s.cache.Files) {
		utilityGained = s.cache.Files[reqFile].Utility(s.cache.CurrentTime, s.cfg)
	}

	// Action a(t) = 1 means cache the requested file
	if action == 1 && !isHit && s.cache.IsValidInsert(reqFile) {
		s.cache.EvictUntilFits(reqFile)
		s.cache.Insert(reqFile)
	}

	// Calculate instant reward r(t) = W(t) - Mem(t) * 100
	reward := s.cache.ComputeReward()

	// Advance physical simulation clock by tau
	s.cache.CurrentTime += tau

	// Sample next arriving request f_r'
	s.currentReq = s.generator.SampleRequestedFile()
	s.cache.RecordRequest(s.currentReq)

	nextState := s.cache.GetCurrentState(s.currentReq)

	return &pb.StepResponse{
		NextState:     mapStateToProto(nextState, s.cache.CurrentTime),
		Reward:        reward,
		Tau:           tau,
		IsHit:         isHit,
		UtilityGained: utilityGained,
	}
}

// mapStateToProto transforms core.State to protobuf StateResponse.
func mapStateToProto(st *core.State, currentTime float64) *pb.StateResponse {
	b32 := make([]int32, len(st.B))
	for i, v := range st.B {
		b32[i] = int32(v)
	}

	return &pb.StateResponse{
		Mem:           st.Mem,
		D:             st.D,
		Y:             st.Y,
		Z:             st.Z,
		B:             b32,
		RequestedFile: int32(st.RequestedFile),
		CurrentTime:   currentTime,
	}
}
