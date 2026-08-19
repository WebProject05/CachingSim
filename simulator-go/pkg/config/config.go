package config

type Config struct {
	TotalFileTypes int     // F = 50
	CacheCapacity  float64 // M = 10000
	SlidingWindowN int     // N = 100
	UTMax          float64 // UT_max = 1.5
	UTMin          float64 // UT_min = 0.1
	Curve          float64 // (UT_max - UT_min) / (e - 1)
	DiscountGamma  float64 // gamma = 0.99
	LambdaSource   float64 // lambda_S = 0.2
	LambdaTarget   float64 // lambda_T = 0.3
	ZipfEta        float64 // eta in [0, 1]
}

func DefaultConfig() *Config {
	e := 2.718281828459045
	utMax := 1.5
	utMin := 0.1
	return &Config{
		TotalFileTypes: 50,
		CacheCapacity:  10000.0,
		SlidingWindowN: 100,
		UTMax:          utMax,
		UTMin:          utMin,
		Curve:          (utMax - utMin) / (e - 1.0),
		DiscountGamma:  0.99,
		LambdaSource:   0.2,
		LambdaTarget:   0.3,
		ZipfEta:        1.0,
	}
}