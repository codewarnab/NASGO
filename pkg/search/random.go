package search

import (
	"context"
	"math/rand"
	"sync"
	"time"

	"nas-go/pkg/searchspace"
)

// RandomSearch implements the Searcher interface using random sampling.
// It's the simplest search strategy: just sample random architectures
// from the search space and evaluate them.
//
// Despite its simplicity, random search is a surprisingly strong baseline.
// Many complex NAS methods only marginally outperform random search!
// This is because:
// 1. Search spaces are often well-designed, making most architectures decent
// 2. Evaluation noise can mask small fitness differences
// 3. Random search has perfect exploration (no exploitation)
//
// Use random search as a baseline to ensure your fancier methods actually help.
//
// Reference: "Random Search and Reproducibility for Neural Architecture Search"
// https://arxiv.org/abs/1902.07638
type RandomSearch struct {
	// rng is the random number generator for reproducibility
	rng *rand.Rand
}

// NewRandomSearch creates a new random search strategy.
//
// Parameters:
//   - seed: Random seed for reproducibility. Use -1 for random seed.
//
// Example:
//
//	searcher := NewRandomSearch(42)
//	result, err := searcher.Search(ctx, config)
func NewRandomSearch(seed int64) *RandomSearch {
	if seed == -1 {
		seed = time.Now().UnixNano()
	}
	return &RandomSearch{
		rng: rand.New(rand.NewSource(seed)),
	}
}

// Name returns the name of this search strategy.
func (r *RandomSearch) Name() string {
	return "RandomSearch"
}

// Search runs random search on the given search space.
// It samples MaxEvaluations random architectures and returns the best one.
//
// The search:
// 1. Samples a random architecture from the search space
// 2. Evaluates it using the EvaluatorFunc
// 3. Tracks the best architecture seen
// 4. Repeats until budget exhausted or context cancelled
//
// Parameters:
//   - ctx: Context for cancellation (e.g., from signal.NotifyContext)
//   - config: Search configuration including budget and evaluator
//
// Returns:
//   - SearchResult with best architecture and history
//   - Error if evaluation fails or context is cancelled
func (r *RandomSearch) Search(ctx context.Context, config SearchConfig) (*SearchResult, error) {
	startTime := time.Now()
	if config.Seed != -1 {
		config.SearchSpace.SetSeed(config.Seed)
	}
	if config.ResumeSearchSpaceRNG != 0 {
		config.SearchSpace.SetRNGState(config.ResumeSearchSpaceRNG)
	}
	result := &SearchResult{History: append([]*searchspace.Architecture(nil), config.ResumeHistory...), StrategyName: r.Name()}
	bestFitness := -1e9
	var bestArch *searchspace.Architecture
	for _, arch := range result.History {
		if arch.Metadata.Fitness > bestFitness {
			bestFitness = arch.Metadata.Fitness
			bestArch = arch
		}
	}
	workers := config.NumWorkers
	if workers < 1 {
		workers = 1
	}

	for offset := len(result.History); offset < config.MaxEvaluations; {
		select {
		case <-ctx.Done():
			result.Cancelled = true
			result.BestArchitecture = bestArch
			result.BestFitness = bestFitness
			result.TotalEvaluations = len(result.History)
			result.SearchDuration = time.Since(startTime)
			return result, ctx.Err()
		default:
		}
		batchSize := workers
		if remaining := config.MaxEvaluations - offset; batchSize > remaining {
			batchSize = remaining
		}
		batch := make([]*searchspace.Architecture, batchSize)
		for i := range batch {
			batch[i] = config.SearchSpace.SampleRandomArchitecture()
			batch[i].Metadata.Generation = 0
		}
		type outcome struct {
			index    int
			fitness  float64
			duration time.Duration
			err      error
		}
		outcomes := make(chan outcome, batchSize)
		var wg sync.WaitGroup
		for i, arch := range batch {
			wg.Add(1)
			go func(i int, arch *searchspace.Architecture) {
				defer wg.Done()
				began := time.Now()
				fitness := -float64(arch.ParameterEstimate())
				var err error
				if config.EvaluatorFunc != nil {
					fitness, err = config.EvaluatorFunc(ctx, arch)
				}
				outcomes <- outcome{i, fitness, time.Since(began), err}
			}(i, arch)
		}
		wg.Wait()
		close(outcomes)
		ordered := make([]outcome, batchSize)
		for o := range outcomes {
			ordered[o.index] = o
		}
		lastSuccessful := -1
		for i, o := range ordered {
			if o.err == nil {
				lastSuccessful = i
			}
		}
		for i, o := range ordered {
			if o.err != nil {
				continue
			}
			arch := batch[i]
			arch.Metadata.Fitness = o.fitness
			arch.Metadata.EvaluationTime = o.duration
			result.History = append(result.History, arch)
			if o.fitness > bestFitness {
				bestFitness = o.fitness
				bestArch = arch
			}
			if config.OnEvaluation != nil {
				config.OnEvaluation(EvaluationEvent{Architecture: arch, Fitness: o.fitness, EvaluationNumber: len(result.History), TotalEvaluations: config.MaxEvaluations, Duration: o.duration, BestSoFar: bestFitness, Generation: 0, SearchSpaceRNG: config.SearchSpace.RNGState(), CheckpointSafe: i == lastSuccessful})
			}
		}
		offset += batchSize
	}
	result.BestArchitecture = bestArch
	result.BestFitness = bestFitness
	result.TotalEvaluations = len(result.History)
	result.SearchDuration = time.Since(startTime)
	return result, nil
}
