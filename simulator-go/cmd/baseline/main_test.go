package main

import (
	"math/rand"
	"reflect"
	"testing"

	"smdp-edge-caching-framework/pkg/config"
	"smdp-edge-caching-framework/pkg/core"
)

func TestRunAllConcurrentMatchesSequential(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.TotalFileTypes = 8
	cfg.CacheCapacity = 100
	files := core.GenerateFiles(cfg, rand.New(rand.NewSource(42)))
	events := generateRequestEvents(cfg, 42, 200)
	sequential := runAllWithConcurrency(cfg, files, events, 20, 10, false)
	concurrent := runAllWithConcurrency(cfg, files, events, 20, 10, true)

	for index := range sequential {
		sequential[index].elapsedSeconds = 0
		concurrent[index].elapsedSeconds = 0
	}
	if !reflect.DeepEqual(sequential, concurrent) {
		t.Fatalf("concurrent results differ from sequential results:\nsequential=%+v\nconcurrent=%+v", sequential, concurrent)
	}
}
