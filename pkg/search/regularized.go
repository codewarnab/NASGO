package search

import (
	"context"
	"math/rand"
	"time"

	"nas-go/pkg/searchspace"
)

// RegularizedEvolution implements the regularized evolution algorithm.
// This is the algorithm from Google Research that achieved state-of-the-art
// results on image classification (AmoebaNet).
//
// The key innovation is the AGING MECHANISM:
// - Each individual has an "age" (time in population)
// - The OLDEST individual is removed, not the worst
// - This prevents "super-individuals" from dominating forever
// - Encourages continuous exploration of new regions
//
// Algorithm steps (each iteration):
// 1. SAMPLE: Pick random subset of population (tournament)
// 2. SELECT: Choose the best individual from sample
// 3. MUTATE: Create child by mutating the selected parent
// 4. EVALUATE: Get fitness of the child
// 5. REMOVE OLDEST: Remove the oldest individual from population
// 6. ADD CHILD: Add the new child to population
//
// Why does aging help?
// - Old individuals may have high fitness due to luck, not quality
// - Removing old individuals ensures fresh blood
// - Population stays diverse, reducing local optima risk
// - Empirically outperforms standard evolution on NAS tasks
//
// Reference: Real et al., "Regularized Evolution for Image Classifier Architecture Search"
// https://arxiv.org/abs/1802.01548
type RegularizedEvolution struct {
	rng *rand.Rand
}

// NewRegularizedEvolution creates a new regularized evolution strategy.
//
// Parameters:
//   - seed: Random seed for reproducibility
func NewRegularizedEvolution(seed int64) *RegularizedEvolution {
	if seed == -1 {
		seed = time.Now().UnixNano()
	}
	return &RegularizedEvolution{
		rng: rand.New(rand.NewSource(seed)),
	}
}

// Name returns the strategy name.
func (r *RegularizedEvolution) Name() string {
	return "RegularizedEvolution"
}

// individual wraps an architecture with its age for the aging mechanism.
// We track when each individual was added to compute relative age.
type individual struct {
	arch    *searchspace.Architecture
	addedAt int // "tick" when this individual was added
	fitness float64
}

// Search runs regularized evolution on the given search space.
//
// The algorithm maintains a fixed-size population. Each step:
// 1. Tournament selection from random sample
// 2. Mutate winner to create child
// 3. Evaluate child
// 4. Remove oldest (not worst!) individual
// 5. Add child to population
//
// This aging mechanism is what makes it "regularized" - it prevents
// any single individual from dominating the population forever.
func (r *RegularizedEvolution) Search(ctx context.Context, config SearchConfig) (*SearchResult, error) {
	startTime := time.Now()

	// Seed the search space
	if config.Seed != -1 {
		config.SearchSpace.SetSeed(config.Seed)
	}

	result := &SearchResult{
		History:      make([]*searchspace.Architecture, 0, config.MaxEvaluations),
		StrategyName: r.Name(),
	}

	// Initialize population as a slice (FIFO for aging)
	// We use a slice instead of a queue structure for simplicity
	// The oldest individual is at index 0, newest at the end
	population := make([]*individual, 0, config.PopulationSize)

	evaluationCount := 0
	var bestFitness float64 = -1e9
	var bestArch *searchspace.Architecture
	tick := 0 // Monotonic counter for aging

	// Phase 1: Initialize population with random architectures
	for len(population) < config.PopulationSize && evaluationCount < config.MaxEvaluations {
		// Check for cancellation
		select {
		case <-ctx.Done():
			return r.buildResult(result, bestArch, bestFitness, evaluationCount, 0, startTime, true), ctx.Err()
		default:
		}

		arch := config.SearchSpace.SampleRandomArchitecture()
		arch.Metadata.Generation = 0

		fitness, err := r.evaluateArch(ctx, config, arch)
		if err != nil {
			continue
		}

		evaluationCount++
		result.History = append(result.History, arch)

		// Add to population
		ind := &individual{
			arch:    arch,
			addedAt: tick,
			fitness: fitness,
		}
		population = append(population, ind)
		tick++

		// Track best
		if fitness > bestFitness {
			bestFitness = fitness
			bestArch = arch
		}

		if config.OnEvaluation != nil {
			config.OnEvaluation(EvaluationEvent{
				Architecture:     arch,
				Fitness:          fitness,
				EvaluationNumber: evaluationCount,
				TotalEvaluations: config.MaxEvaluations,
				Duration:         arch.Metadata.EvaluationTime,
				BestSoFar:        bestFitness,
				Generation:       0,
			})
		}
	}

	// Phase 2: Evolution with aging
	generation := 1
	for evaluationCount < config.MaxEvaluations {
		// Check for cancellation
		select {
		case <-ctx.Done():
			return r.buildResult(result, bestArch, bestFitness, evaluationCount, generation, startTime, true), ctx.Err()
		default:
		}

		// === TOURNAMENT SELECTION ===
		// Sample random subset of population
		sampleSize := config.TournamentSize
		if sampleSize > len(population) {
			sampleSize = len(population)
		}
		if sampleSize < 1 {
			sampleSize = 1
		}

		// Random sample without replacement
		sample := r.randomSample(population, sampleSize)

		// Select best from sample (tournament winner)
		parent := r.selectBest(sample)

		// === MUTATION ===
		child := config.SearchSpace.Mutate(parent.arch)
		child.Metadata.Generation = generation

		// === EVALUATION ===
		fitness, err := r.evaluateArch(ctx, config, child)
		if err != nil {
			continue
		}

		evaluationCount++
		result.History = append(result.History, child)

		// === AGING: REMOVE OLDEST ===
		// This is the key innovation of regularized evolution!
		// We remove the oldest individual, NOT the worst.
		// The oldest is at index 0 (FIFO order).
		if len(population) >= config.PopulationSize {
			population = population[1:] // Remove oldest (front of slice)
		}

		// === ADD CHILD ===
		childInd := &individual{
			arch:    child,
			addedAt: tick,
			fitness: fitness,
		}
		population = append(population, childInd)
		tick++

		// Track best overall (not just in population)
		if fitness > bestFitness {
			bestFitness = fitness
			bestArch = child
		}

		if config.OnEvaluation != nil {
			config.OnEvaluation(EvaluationEvent{
				Architecture:     child,
				Fitness:          fitness,
				EvaluationNumber: evaluationCount,
				TotalEvaluations: config.MaxEvaluations,
				Duration:         child.Metadata.EvaluationTime,
				BestSoFar:        bestFitness,
				Generation:       generation,
			})
		}

		generation++
	}

	return r.buildResult(result, bestArch, bestFitness, evaluationCount, generation, startTime, false), nil
}

// randomSample returns a random sample of size k from the population.
// Sampling is without replacement.
func (r *RegularizedEvolution) randomSample(population []*individual, k int) []*individual {
	// Fisher-Yates partial shuffle
	n := len(population)
	if k > n {
		k = n
	}

	// Create index array
	indices := make([]int, n)
	for i := range indices {
		indices[i] = i
	}

	// Partial shuffle: only shuffle first k elements
	for i := 0; i < k; i++ {
		j := i + r.rng.Intn(n-i)
		indices[i], indices[j] = indices[j], indices[i]
	}

	// Extract sample
	sample := make([]*individual, k)
	for i := 0; i < k; i++ {
		sample[i] = population[indices[i]]
	}

	return sample
}

// selectBest returns the individual with highest fitness from the sample.
// This is the tournament selection step.
func (r *RegularizedEvolution) selectBest(sample []*individual) *individual {
	if len(sample) == 0 {
		return nil
	}

	best := sample[0]
	for _, ind := range sample[1:] {
		if ind.fitness > best.fitness {
			best = ind
		}
	}
	return best
}

// evaluateArch evaluates an architecture and updates its metadata.
func (r *RegularizedEvolution) evaluateArch(
	ctx context.Context,
	config SearchConfig,
	arch *searchspace.Architecture,
) (float64, error) {
	evalStart := time.Now()
	var fitness float64
	var err error

	if config.EvaluatorFunc != nil {
		fitness, err = config.EvaluatorFunc(ctx, arch)
		if err != nil {
			return 0, err
		}
	} else {
		// Default: use negative parameter count as proxy
		fitness = -float64(arch.ParameterEstimate())
	}

	arch.Metadata.Fitness = fitness
	arch.Metadata.EvaluationTime = time.Since(evalStart)
	return fitness, nil
}

// buildResult constructs the final SearchResult.
func (r *RegularizedEvolution) buildResult(
	result *SearchResult,
	bestArch *searchspace.Architecture,
	bestFitness float64,
	evaluationCount int,
	generation int,
	startTime time.Time,
	cancelled bool,
) *SearchResult {
	result.BestArchitecture = bestArch
	result.BestFitness = bestFitness
	result.TotalEvaluations = evaluationCount
	result.FinalGeneration = generation
	result.SearchDuration = time.Since(startTime)
	result.Cancelled = cancelled
	return result
}
