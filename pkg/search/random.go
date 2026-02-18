package search

import (
	"context"
	"math/rand"
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

	// Apply seed to search space for reproducible sampling
	if config.Seed != -1 {
		config.SearchSpace.SetSeed(config.Seed)
	}

	// Initialize result tracking
	result := &SearchResult{
		History:      make([]*searchspace.Architecture, 0, config.MaxEvaluations),
		StrategyName: r.Name(),
	}

	var bestFitness float64 = -1e9 // Start with very low value
	var bestArch *searchspace.Architecture

	// Main search loop
	for i := 0; i < config.MaxEvaluations; i++ {
		// Check for cancellation before each evaluation
		// This is the graceful shutdown pattern - check ctx.Done() regularly
		select {
		case <-ctx.Done():
			// Context was cancelled (e.g., user pressed Ctrl+C)
			result.Cancelled = true
			result.BestArchitecture = bestArch
			result.BestFitness = bestFitness
			result.TotalEvaluations = i
			result.SearchDuration = time.Since(startTime)
			return result, ctx.Err()
		default:
			// Continue with search
		}

		// Sample a random architecture
		arch := config.SearchSpace.SampleRandomArchitecture()
		arch.Metadata.Generation = 0 // Random search has no generations

		// Evaluate the architecture
		evalStart := time.Now()
		var fitness float64
		var err error

		if config.EvaluatorFunc != nil {
			fitness, err = config.EvaluatorFunc(ctx, arch)
			if err != nil {
				// If evaluation fails, skip this architecture
				// In production, you might want more sophisticated error handling
				continue
			}
		} else {
			// No evaluator provided - use parameter count as proxy
			// (fewer parameters = higher fitness in this simple case)
			fitness = -float64(arch.ParameterEstimate())
		}

		// Record fitness
		arch.Metadata.Fitness = fitness
		arch.Metadata.EvaluationTime = time.Since(evalStart)
		result.History = append(result.History, arch)

		// Update best if this is better
		if fitness > bestFitness {
			bestFitness = fitness
			bestArch = arch
		}

		// Call evaluation callback if provided
		if config.OnEvaluation != nil {
			config.OnEvaluation(EvaluationEvent{
				Architecture:     arch,
				Fitness:          fitness,
				EvaluationNumber: i + 1,
				TotalEvaluations: config.MaxEvaluations,
				Duration:         arch.Metadata.EvaluationTime,
				BestSoFar:        bestFitness,
				Generation:       0,
			})
		}
	}

	// Finalize result
	result.BestArchitecture = bestArch
	result.BestFitness = bestFitness
	result.TotalEvaluations = len(result.History)
	result.SearchDuration = time.Since(startTime)

	return result, nil
}
