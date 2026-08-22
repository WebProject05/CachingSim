package server

import (
	"context"
	"math/rand"
	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
	"smdp-edge-caching-framework/pkg/pb"
	"smdp-edge-caching-framework/pkg/smdp"
)

type EnvServer struct {
	pb.UnimplementedCacheEnvServiceServer
	cfg        *config.Config
	cache      *core.CacheEngine
	generator  *smdp.Generator
	currentReq int
}

func NewEnvServer(cfg *config.Config, seed int64) *EnvServer {
	rng := rand.New(rand.NewSource(seed))
	files := core.GenerateFiles(cfg, rng)
	cache := core.NewCacheEngine(cfg, files)
	generator := smdp.NewGenerator(cfg, seed)

	return &EnvServer{
		cfg:        cfg,
		cache:      cache,
		generator:  generator,
		currentReq: generator.SampleRequestedFile(),
	}
}

func (s *EnvServer) Reset(ctx context.Context, req *pb.ResetRequest) (*pb.StateResponse, error) {
	if req.Lambda > 0 {
		s.cfg.LambdaSource = req.Lambda
	}
	if req.Eta > 0 {
		s.cfg.ZipfEta = req.Eta
	}

	seed := req.Seed
	if seed == 0 {
		seed = 42
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

func (s *EnvServer) Step(ctx context.Context, req *pb.StepRequest) (*pb.StepResponse, error) {
	return s.executeStep(req.Action, s.generator.NextPoissonInterval(s.cfg.LambdaSource)), nil
}

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

// BatchStep processes multiple actions in sequence and returns a batch of responses
func (s *EnvServer) BatchStep(ctx context.Context, req *pb.BatchStepRequest) (*pb.BatchStepResponse, error) {
	responses := make([]*pb.StepResponse, len(req.Actions))
	for i, action := range req.Actions {
		responses[i] = s.executeStep(action, s.generator.NextPoissonInterval(s.cfg.LambdaSource))
	}
	return &pb.BatchStepResponse{StepResponses: responses}, nil
}

// MDPStep simulates a discrete-time MDP transition using a fixed time quota
// instead of a stochastic Poisson inter-arrival time.
func (s *EnvServer) MDPStep(action int32, timeQuota float64) *pb.StepResponse {
	return s.executeStep(action, timeQuota)
}

func (s *EnvServer) executeStep(action int32, tau float64) *pb.StepResponse {
	isHit := s.cache.IsCached(s.currentReq)
	var utilityGained float64
	if isHit {
		utilityGained = s.cache.Files[s.currentReq].Utility(s.cache.CurrentTime, s.cfg)
	}

	if action == 1 && !isHit && s.cache.IsValidInsert(s.currentReq) {
		s.cache.EvictUntilFits(s.currentReq)
		s.cache.Insert(s.currentReq)
	}

	reward := s.cache.ComputeReward()
	s.cache.CurrentTime += tau
	s.currentReq = s.generator.SampleRequestedFile()
	s.cache.RecordRequest(s.currentReq)

	return &pb.StepResponse{
		NextState:     mapStateToProto(s.cache.GetCurrentState(s.currentReq), s.cache.CurrentTime),
		Reward:        reward,
		Tau:           tau,
		IsHit:         isHit,
		UtilityGained: utilityGained,
	}
}
