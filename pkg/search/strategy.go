// Package search implements various neural architecture search strategies.
// Each strategy explores the search space differently:
// - RandomSearch: Simple random sampling (baseline)
// - EvolutionarySearch: Population-based mutation and selection
// - RegularizedEvolution: Evolution with aging to prevent stagnation
//
// All strategies implement the Searcher interface for interchangeability.
package search

import (
	"context"
	"fmt"
	"sync"
	"time"

	"nas-go/pkg/searchspace"
)

// Searcher is the interface that all search strategies must implement.
// This allows swapping strategies without changing the calling code.
//
// Why an interface?
// 1. Testability: Easy to mock for unit tests
// 2. Extensibility: Add new strategies without modifying existing code
// 3. Flexibility: Choose strategy at runtime based on config
type Searcher interface {
	// Search runs the architecture search with the given configuration.
	// It should respect context cancellation for graceful shutdown.
	//
	// Parameters:
	//   - ctx: Context for cancellation (from signal.NotifyContext or timeout)
	//   - config: Search configuration (budget, population size, etc.)
	//
	// Returns:
	//   - SearchResult containing best architecture and search history
	//   - Error if search fails or is cancelled
	Search(ctx context.Context, config SearchConfig) (*SearchResult, error)

	// Name returns the human-readable name of this search strategy.
	// Used for logging and experiment tracking.
	Name() string
}

// SearchConfig holds all configuration for a search run.
// Using a struct for config (rather than many parameters) makes it:
// 1. Easy to add new options without breaking existing code
// 2. Easy to serialize/deserialize for experiment tracking
// 3. Self-documenting with field names
type SearchConfig struct {
	// SearchSpace defines what architectures are valid.
	SearchSpace *searchspace.SearchSpace `json:"search_space"`

	// MaxEvaluations is the total evaluation budget.
	// Search stops when this many architectures have been evaluated.
	// Typical values: 1000-10000 for proxy evaluation, 100-500 for full training.
	MaxEvaluations int `json:"max_evaluations"`

	// PopulationSize is used by evolutionary strategies.
	// Ignored by random search.
	// Typical values: 50-200. Larger = more diversity, slower convergence.
	PopulationSize int `json:"population_size"`

	// TournamentSize is the sample size for tournament selection.
	// Used by regularized evolution. Typically 10-50% of population.
	TournamentSize int `json:"tournament_size"`

	// NumWorkers is the number of parallel evaluation workers.
	// Set to 0 or 1 for sequential evaluation.
	// Set to runtime.NumCPU() for maximum parallelism.
	NumWorkers int `json:"num_workers"`

	// Seed for random number generation.
	// Set to a fixed value for reproducibility.
	// Set to -1 for random seed.
	Seed int64 `json:"seed"`

	// EvaluatorFunc is the function used to evaluate architectures.
	// This allows plugging in different evaluation methods (proxy, full training).
	// If nil, architectures will have fitness = 0 (useful for testing search logic).
	EvaluatorFunc EvaluatorFunc `json:"-"` // Excluded from JSON

	// OnEvaluation is called after each architecture is evaluated.
	// Useful for logging, checkpointing, or early stopping.
	// Can be nil.
	OnEvaluation EvaluationCallback `json:"-"`
}

// EvaluatorFunc is a function type for evaluating architectures.
// It takes an architecture and returns a fitness score (higher is better).
//
// Parameters:
//   - ctx: Context for cancellation
//   - arch: Architecture to evaluate
//
// Returns:
//   - fitness: Score indicating how good this architecture is (higher = better)
//   - error: If evaluation fails
type EvaluatorFunc func(ctx context.Context, arch *searchspace.Architecture) (float64, error)

// EvaluationCallback is called after each architecture is evaluated.
// Useful for progress tracking, logging, and checkpointing.
type EvaluationCallback func(eval EvaluationEvent)

// EvaluationEvent contains information about a single evaluation.
type EvaluationEvent struct {
	// Architecture that was evaluated
	Architecture *searchspace.Architecture

	// Fitness score achieved
	Fitness float64

	// EvaluationNumber is which evaluation this is (1-indexed)
	EvaluationNumber int

	// TotalEvaluations is the budget
	TotalEvaluations int

	// Duration of this evaluation
	Duration time.Duration

	// BestSoFar is the best fitness seen so far
	BestSoFar float64

	// Generation for evolutionary methods (0 for random)
	Generation int
}

// SearchResult contains the outcome of a search run.
type SearchResult struct {
	// BestArchitecture is the architecture with highest fitness found.
	BestArchitecture *searchspace.Architecture `json:"best_architecture"`

	// BestFitness is the fitness of the best architecture.
	BestFitness float64 `json:"best_fitness"`

	// History contains all evaluated architectures in order.
	// Can be large - consider limiting for memory in production.
	History []*searchspace.Architecture `json:"history,omitempty"`

	// TotalEvaluations is how many architectures were evaluated.
	TotalEvaluations int `json:"total_evaluations"`

	// SearchDuration is the total time spent searching.
	SearchDuration time.Duration `json:"search_duration"`

	// FinalGeneration for evolutionary methods.
	FinalGeneration int `json:"final_generation,omitempty"`

	// StrategyName identifies which strategy was used.
	StrategyName string `json:"strategy_name"`

	// Cancelled is true if search was stopped early by context cancellation.
	Cancelled bool `json:"cancelled"`
}

// Summary returns a human-readable summary of the search result.
func (r *SearchResult) Summary() string {
	status := "completed"
	if r.Cancelled {
		status = "cancelled"
	}
	return fmt.Sprintf(
		"Search %s (%s):\n"+
			"  Best Fitness:    %.4f\n"+
			"  Evaluations:     %d\n"+
			"  Duration:        %s\n"+
			"  Best Arch ID:    %s",
		status, r.StrategyName,
		r.BestFitness,
		r.TotalEvaluations,
		r.SearchDuration.Round(time.Millisecond),
		r.BestArchitecture.ID[:8],
	)
}

// DefaultSearchConfig returns a sensible default configuration.
func DefaultSearchConfig(space *searchspace.SearchSpace) SearchConfig {
	return SearchConfig{
		SearchSpace:    space,
		MaxEvaluations: 1000,
		PopulationSize: 100,
		TournamentSize: 25,
		NumWorkers:     1,
		Seed:           42,
	}
}

// workerPool manages parallel architecture evaluation.
// Uses a producer-consumer pattern with worker goroutines.
type workerPool struct {
	numWorkers    int
	evaluatorFunc EvaluatorFunc
	wg            sync.WaitGroup
}

// newWorkerPool creates a new worker pool.
func newWorkerPool(numWorkers int, evaluatorFunc EvaluatorFunc) *workerPool {
	if numWorkers < 1 {
		numWorkers = 1
	}
	return &workerPool{
		numWorkers:    numWorkers,
		evaluatorFunc: evaluatorFunc,
	}
}

// evaluationJob is a unit of work for the worker pool.
type evaluationJob struct {
	arch   *searchspace.Architecture
	result chan evaluationResult
}

// evaluationResult is the output of an evaluation.
type evaluationResult struct {
	arch    *searchspace.Architecture
	fitness float64
	err     error
}
