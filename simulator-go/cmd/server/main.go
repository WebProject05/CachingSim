package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/grpc"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
	"smdp-edge-caching-framework/pkg/pb"
	"smdp-edge-caching-framework/pkg/server"
	"smdp-edge-caching-framework/pkg/smdp"
)

func main() {
	port := flag.Int("port", 50051, "gRPC server listening port")
	isStandalone := flag.Bool("standalone", false, "run internal standalone simulation benchmark instead of gRPC server")
	seed := flag.Int64("seed", 42, "random seed")
	totalTrials := flag.Int("trials", 1000, "number of trials for standalone simulation")
	fileCount := flag.Int("files", 50, "number of file types F")
	capacity := flag.Float64("capacity", 10000, "cache capacity in MiB")
	lambda := flag.Float64("lambda", 0.2, "Poisson request rate lambda")
	eta := flag.Float64("eta", 1.0, "Zipf skewness eta")
	flag.Parse()

	cfg := config.DefaultConfig()
	cfg.TotalFileTypes = *fileCount
	cfg.CacheCapacity = *capacity
	cfg.LambdaSource = *lambda
	cfg.ZipfEta = *eta

	if *isStandalone {
		runStandaloneSimulation(cfg, *seed, *totalTrials)
		return
	}

	// Start live gRPC server
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen on port %d: %v", *port, err)
	}

	grpcServer := grpc.NewServer()
	envServer := server.NewEnvServer(cfg, *seed)
	pb.RegisterCacheEnvServiceServer(grpcServer, envServer)

	// Graceful shutdown channel
	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("=== SMDP Edge Caching gRPC Server listening on port %d ===", *port)
		log.Printf("Configuration: Files=%d, Capacity=%.0f MiB, Lambda=%.2f, ZipfEta=%.2f",
			cfg.TotalFileTypes, cfg.CacheCapacity, cfg.LambdaSource, cfg.ZipfEta)
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("gRPC server error: %v", err)
		}
	}()

	<-stopChan
	log.Println("Shutting down gRPC server gracefully...")
	grpcServer.GracefulStop()
	log.Println("Server stopped.")
}

func runStandaloneSimulation(cfg *config.Config, seed int64, totalTrials int) {
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(seed))

	// Initialize System Components
	files := core.GenerateFiles(cfg, rng)
	cache := core.NewCacheEngine(cfg, files)
	generator := smdp.NewGenerator(cfg, seed)

	hitCount := 0
	totalUtility := 0.0
	totalReward := 0.0

	fmt.Println("=== Starting SMDP Edge Caching Standalone Simulation (Go Native Engine) ===")
	fmt.Printf("File Types: %d, Capacity: %.0f MiB, Lambda: %.2f, ZipfEta: %.2f, Seed: %d\n\n",
		cfg.TotalFileTypes, cfg.CacheCapacity, cfg.LambdaSource, cfg.ZipfEta, seed)

	for trial := 1; trial <= totalTrials; trial++ {
		// 1. Generate Poisson arrival interval tau and advance time
		tau := generator.NextPoissonInterval(cfg.LambdaSource)
		cache.CurrentTime += tau

		// 2. Sample incoming request f_r via Zipf distribution
		reqFile := generator.SampleRequestedFile()
		cache.RecordRequest(reqFile)

		// 3. Check Cache Hit
		isHit := cache.IsCached(reqFile)
		if isHit {
			hitCount++
			totalUtility += files[reqFile].Utility(cache.CurrentTime, cfg)
		}

		// 4. Decision: Reactive caching with lowest-utility eviction
		action := 1
		if action == 1 && !isHit {
			cache.EvictUntilFits(reqFile)
			cache.Insert(reqFile)
		}

		// 5. Compute Reward r(t)
		r := cache.ComputeReward()
		totalReward += r

		if trial%200 == 0 || trial == totalTrials {
			state := cache.GetCurrentState(reqFile)
			fmt.Printf("Trial [%4d/%4d] | Hits: %3d | Unused Mem: %5.1f%% | Instant Reward: %7.2f | Avg Reward: %7.2f\n",
				trial, totalTrials, hitCount, state.Mem*100.0, r, totalReward/float64(trial))
		}
	}

	fmt.Println("\n=== Final Simulation Metrics ===")
	fmt.Printf("Total Hits: %d / %d (%.2f%%)\n", hitCount, totalTrials, float64(hitCount)/float64(totalTrials)*100.0)
	fmt.Printf("Accumulated Total Utility: %.2f\n", totalUtility)
	fmt.Printf("Average Reward in Long Run: %.2f\n", totalReward/float64(totalTrials))
}
