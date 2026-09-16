package search

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"time"

	"nas-go/pkg/rngstate"
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
	rng       *rand.Rand
	rngSource *rngstate.Source
}

// NewEvolutionarySearch creates a new evolutionary search strategy.
//
// Parameters:
//   - seed: Random seed for reproducibility
func NewEvolutionarySearch(seed int64) *EvolutionarySearch {
	if seed == -1 {
		seed = time.Now().UnixNano()
	}
	source := rngstate.New(seed)
	return &EvolutionarySearch{rng: rand.New(source), rngSource: source}
}

// Name returns the strategy name.
func (e *EvolutionarySearch) Name() string {
	return "EvolutionarySearch"
}

// Search runs evolutionary search. Initial candidates and offspring are
// evaluated in bounded batches while state updates remain deterministic.
func (e *EvolutionarySearch) Search(ctx context.Context, config SearchConfig) (*SearchResult, error) {
	startTime := time.Now()
	if config.Seed != -1 {
		config.SearchSpace.SetSeed(config.Seed)
	}
	result := &SearchResult{History: append([]*searchspace.Architecture(nil), config.ResumeHistory...), StrategyName: e.Name()}
	population := append([]*searchspace.Architecture(nil), config.ResumePopulation...)
	if len(population) == 0 {
		population = append(population, config.ResumeHistory...)
	}
	if config.ResumeStrategyRNG != 0 {
		e.rngSource.State = config.ResumeStrategyRNG
	}
	if config.ResumeSearchSpaceRNG != 0 {
		config.SearchSpace.SetRNGState(config.ResumeSearchSpaceRNG)
	}
	if len(population) > config.PopulationSize {
		population = population[len(population)-config.PopulationSize:]
	}
	bestFitness := -1e9
	var bestArch *searchspace.Architecture
	for _, arch := range result.History {
		if arch.Metadata.Fitness > bestFitness {
			bestFitness, bestArch = arch.Metadata.Fitness, arch
		}
	}
	evaluationCount := len(result.History)
	generation := 0

	finish := func(cancelled bool) (*SearchResult, error) {
		result.BestArchitecture, result.BestFitness = bestArch, bestFitness
		result.TotalEvaluations, result.FinalGeneration = evaluationCount, generation
		result.SearchDuration, result.Cancelled = time.Since(startTime), cancelled
		if cancelled {
			return result, ctx.Err()
		}
		return result, nil
	}
	commit := func(o batchOutcome, gen int) {
		o.arch.Metadata.Fitness, o.arch.Metadata.EvaluationTime = o.fitness, o.duration
		evaluationCount++
		result.History = append(result.History, o.arch)
		if o.fitness > bestFitness {
			bestFitness, bestArch = o.fitness, o.arch
		}
	}

	missing := config.PopulationSize - len(population)
	if remaining := config.MaxEvaluations - evaluationCount; missing > remaining {
		missing = remaining
	}
	failedInitializations := 0
	maxFailedInitializations := config.PopulationSize * 3
	if maxFailedInitializations < config.NumWorkers {
		maxFailedInitializations = config.NumWorkers
	}
	for missing > 0 {
		if ctx.Err() != nil {
			return finish(true)
		}
		n := config.NumWorkers
		if n < 1 {
			n = 1
		}
		if n > missing {
			n = missing
		}
		batch := config.SearchSpace.PopulateInitial(n)
		for _, o := range evaluateBatch(ctx, config.NumWorkers, batch, func(c context.Context, a *searchspace.Architecture) (float64, error) {
			return e.evaluateArch(c, config, a)
		}) {
			if o.err != nil || o.arch == nil {
				failedInitializations++
				continue
			}
			population = append(population, o.arch)
			commit(o, 0)
		}
		if failedInitializations >= maxFailedInitializations && len(population) < config.PopulationSize {
			return nil, fmt.Errorf("initializing population: evaluator failed %d candidates without filling population", failedInitializations)
		}
		missing = config.PopulationSize - len(population)
		if remaining := config.MaxEvaluations - evaluationCount; missing > remaining {
			missing = remaining
		}
	}
	if len(population) == 0 {
		return finish(ctx.Err() != nil)
	}
	generation = 1
	for evaluationCount < config.MaxEvaluations {
		if ctx.Err() != nil {
			return finish(true)
		}
		sort.SliceStable(population, func(i, j int) bool { return population[i].Metadata.Fitness > population[j].Metadata.Fitness })
		n := config.NumWorkers
		if n < 1 {
			n = 1
		}
		if rem := config.MaxEvaluations - evaluationCount; n > rem {
			n = rem
		}
		batch := make([]*searchspace.Architecture, n)
		parentPool := (len(population) + 1) / 2
		for i := range batch {
			parent := population[e.rng.Intn(parentPool)]
			batch[i] = config.SearchSpace.Mutate(parent)
			batch[i].Metadata.Generation = generation
		}
		for _, o := range evaluateBatch(ctx, config.NumWorkers, batch, func(c context.Context, a *searchspace.Architecture) (float64, error) {
			return e.evaluateArch(c, config, a)
		}) {
			if o.err != nil || o.arch == nil {
				continue
			}
			commit(o, generation)
			sort.SliceStable(population, func(i, j int) bool { return population[i].Metadata.Fitness > population[j].Metadata.Fitness })
			if len(population) < config.PopulationSize {
				population = append(population, o.arch)
			} else if o.fitness > population[len(population)-1].Metadata.Fitness {
				population[len(population)-1] = o.arch
			}
			if config.OnEvaluation != nil {
				config.OnEvaluation(EvaluationEvent{Architecture: o.arch, Fitness: o.fitness, EvaluationNumber: evaluationCount, TotalEvaluations: config.MaxEvaluations, Duration: o.duration, BestSoFar: bestFitness, Generation: generation, Population: append([]*searchspace.Architecture(nil), population...), StrategyRNG: e.rngSource.State, SearchSpaceRNG: config.SearchSpace.RNGState()})
			}
		}
		generation++
	}
	return finish(false)
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
