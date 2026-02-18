package search

import (
	"context"
	"math/rand"
	"sort"
	"time"

	"nas-go/pkg/searchspace"
)

// EvolutionarySearch implements a basic genetic algorithm for architecture search.
// It maintains a population of architectures and evolves them through:
// 1. Selection: Pick the best individuals
// 2. Mutation: Create variants of selected individuals
// 3. Replacement: Replace worst individuals with mutated offspring
//
// This is a (μ + λ) evolutionary strategy where:
// - μ (mu) = population size (kept between generations)
// - λ (lambda) = number of offspring created each generation
//
// Differences from regularized evolution:
// - No aging mechanism (best architectures can persist forever)
// - Uses fitness-based selection (not tournament + age)
// - Can get stuck in local optima more easily
//
// Use this when:
// - You want to understand how evolution helps
// - Comparison baseline for regularized evolution
// - Fast convergence is more important than exploration
type EvolutionarySearch struct {
	rng *rand.Rand
}

// NewEvolutionarySearch creates a new evolutionary search strategy.
//
// Parameters:
//   - seed: Random seed for reproducibility
func NewEvolutionarySearch(seed int64) *EvolutionarySearch {
	if seed == -1 {
		seed = time.Now().UnixNano()
	}
	return &EvolutionarySearch{
		rng: rand.New(rand.NewSource(seed)),
	}
}

// Name returns the strategy name.
func (e *EvolutionarySearch) Name() string {
	return "EvolutionarySearch"
}

// Search runs evolutionary search on the given search space.
//
// Algorithm:
//  1. Initialize random population
//  2. Evaluate all individuals
//  3. While budget not exhausted:
//     a. Select parents (top individuals)
//     b. Create offspring through mutation
//     c. Evaluate offspring
//     d. Replace worst individuals with offspring
//  4. Return best individual found
func (e *EvolutionarySearch) Search(ctx context.Context, config SearchConfig) (*SearchResult, error) {
	startTime := time.Now()

	// Seed the search space
	if config.Seed != -1 {
		config.SearchSpace.SetSeed(config.Seed)
	}

	result := &SearchResult{
		History:      make([]*searchspace.Architecture, 0, config.MaxEvaluations),
		StrategyName: e.Name(),
	}

	// Initialize population with random architectures
	population := config.SearchSpace.PopulateInitial(config.PopulationSize)

	// Evaluate initial population
	evaluationCount := 0
	var bestFitness float64 = -1e9
	var bestArch *searchspace.Architecture

	for _, arch := range population {
		// Check for cancellation
		select {
		case <-ctx.Done():
			result.Cancelled = true
			result.BestArchitecture = bestArch
			result.BestFitness = bestFitness
			result.TotalEvaluations = evaluationCount
			result.SearchDuration = time.Since(startTime)
			return result, ctx.Err()
		default:
		}

		fitness, err := e.evaluateArch(ctx, config, arch)
		if err != nil {
			continue
		}
		evaluationCount++
		result.History = append(result.History, arch)

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

		if evaluationCount >= config.MaxEvaluations {
			break
		}
	}

	// Evolution loop
	generation := 1
	for evaluationCount < config.MaxEvaluations {
		// Check for cancellation
		select {
		case <-ctx.Done():
			result.Cancelled = true
			result.BestArchitecture = bestArch
			result.BestFitness = bestFitness
			result.TotalEvaluations = evaluationCount
			result.FinalGeneration = generation
			result.SearchDuration = time.Since(startTime)
			return result, ctx.Err()
		default:
		}

		// Sort population by fitness (descending)
		sort.Slice(population, func(i, j int) bool {
			return population[i].Metadata.Fitness > population[j].Metadata.Fitness
		})

		// Select parent from top half of population
		parentIdx := e.rng.Intn(config.PopulationSize / 2)
		parent := population[parentIdx]

		// Create mutated offspring
		offspring := config.SearchSpace.Mutate(parent)
		offspring.Metadata.Generation = generation

		// Evaluate offspring
		evalStart := time.Now()
		fitness, err := e.evaluateArch(ctx, config, offspring)
		if err != nil {
			continue
		}
		offspring.Metadata.EvaluationTime = time.Since(evalStart)
		evaluationCount++
		result.History = append(result.History, offspring)

		// Update best
		if fitness > bestFitness {
			bestFitness = fitness
			bestArch = offspring
		}

		// Replace worst individual with offspring if offspring is better
		// This is elitist selection - we keep the best individuals
		worstIdx := len(population) - 1
		if fitness > population[worstIdx].Metadata.Fitness {
			population[worstIdx] = offspring
		}

		if config.OnEvaluation != nil {
			config.OnEvaluation(EvaluationEvent{
				Architecture:     offspring,
				Fitness:          fitness,
				EvaluationNumber: evaluationCount,
				TotalEvaluations: config.MaxEvaluations,
				Duration:         offspring.Metadata.EvaluationTime,
				BestSoFar:        bestFitness,
				Generation:       generation,
			})
		}

		generation++
	}

	result.BestArchitecture = bestArch
	result.BestFitness = bestFitness
	result.TotalEvaluations = evaluationCount
	result.FinalGeneration = generation
	result.SearchDuration = time.Since(startTime)

	return result, nil
}

// evaluateArch evaluates an architecture and updates its metadata.
func (e *EvolutionarySearch) evaluateArch(
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
		fitness = -float64(arch.ParameterEstimate())
	}

	arch.Metadata.Fitness = fitness
	arch.Metadata.EvaluationTime = time.Since(evalStart)
	return fitness, nil
}
