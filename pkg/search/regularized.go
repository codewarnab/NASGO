package search

import (
	"context"
	"math/rand"
	"time"

	"nas-go/pkg/rngstate"
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
	rng       *rand.Rand
	rngSource *rngstate.Source
}

// NewRegularizedEvolution creates a new regularized evolution strategy.
//
// Parameters:
//   - seed: Random seed for reproducibility
func NewRegularizedEvolution(seed int64) *RegularizedEvolution {
	if seed == -1 {
		seed = time.Now().UnixNano()
	}
	source := rngstate.New(seed)
	return &RegularizedEvolution{rng: rand.New(source), rngSource: source}
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

// Search runs regularized evolution with bounded parallel evaluation batches.
// Selection and population commits remain ordered, preserving aging semantics.
func (r *RegularizedEvolution) Search(ctx context.Context, config SearchConfig) (*SearchResult, error) {
	start := time.Now()
	if config.Seed != -1 {
		config.SearchSpace.SetSeed(config.Seed)
	}
	result := &SearchResult{History: append([]*searchspace.Architecture(nil), config.ResumeHistory...), StrategyName: r.Name()}
	population := make([]*individual, 0, config.PopulationSize)
	for i, a := range config.ResumePopulation {
		population = append(population, &individual{arch: a, addedAt: i, fitness: a.Metadata.Fitness})
	}
	if len(population) == 0 {
		start := len(config.ResumeHistory) - config.PopulationSize
		if start < 0 {
			start = 0
		}
		for i, a := range config.ResumeHistory[start:] {
			population = append(population, &individual{arch: a, addedAt: i, fitness: a.Metadata.Fitness})
		}
	}
	if config.ResumeStrategyRNG != 0 {
		r.rngSource.State = config.ResumeStrategyRNG
	}
	if config.ResumeSearchSpaceRNG != 0 {
		config.SearchSpace.SetRNGState(config.ResumeSearchSpaceRNG)
	}
	count, tick, generation := len(config.ResumeHistory), len(population), 0
	best := -1e9
	var bestArch *searchspace.Architecture
	for _, a := range config.ResumeHistory {
		if a.Metadata.Fitness > best {
			best, bestArch = a.Metadata.Fitness, a
		}
	}
	finish := func(cancelled bool) (*SearchResult, error) {
		out := r.buildResult(result, bestArch, best, count, generation, start, cancelled)
		if cancelled {
			return out, ctx.Err()
		}
		return out, nil
	}
	commit := func(o batchOutcome, gen int) {
		o.arch.Metadata.Fitness, o.arch.Metadata.EvaluationTime = o.fitness, o.duration
		count++
		result.History = append(result.History, o.arch)
		if o.fitness > best {
			best, bestArch = o.fitness, o.arch
		}
	}
	for len(population) < config.PopulationSize && count < config.MaxEvaluations {
		if ctx.Err() != nil {
			return finish(true)
		}
		n := config.NumWorkers
		if n < 1 {
			n = 1
		}
		if rem := config.PopulationSize - len(population); n > rem {
			n = rem
		}
		if rem := config.MaxEvaluations - count; n > rem {
			n = rem
		}
		batch := make([]*searchspace.Architecture, n)
		for i := range batch {
			batch[i] = config.SearchSpace.SampleRandomArchitecture()
		}
		for _, o := range evaluateBatch(ctx, config.NumWorkers, batch, func(c context.Context, a *searchspace.Architecture) (float64, error) {
			return r.evaluateArch(c, config, a)
		}) {
			if o.err != nil || o.arch == nil {
				continue
			}
			commit(o, 0)
			population = append(population, &individual{arch: o.arch, addedAt: tick, fitness: o.fitness})
			tick++
			if config.OnEvaluation != nil {
				pops := make([]*searchspace.Architecture, len(population))
				for i, ind := range population {
					pops[i] = ind.arch
				}
				config.OnEvaluation(EvaluationEvent{Architecture: o.arch, Fitness: o.fitness, EvaluationNumber: count, TotalEvaluations: config.MaxEvaluations, Duration: o.duration, BestSoFar: best, Generation: generation, Population: pops, StrategyRNG: r.rngSource.State, SearchSpaceRNG: config.SearchSpace.RNGState()})
			}
		}
	}
	generation = 1
	for count < config.MaxEvaluations {
		if ctx.Err() != nil {
			return finish(true)
		}
		n := config.NumWorkers
		if n < 1 {
			n = 1
		}
		if rem := config.MaxEvaluations - count; n > rem {
			n = rem
		}
		batch := make([]*searchspace.Architecture, n)
		for i := range batch {
			sampleSize := config.TournamentSize
			if sampleSize > len(population) {
				sampleSize = len(population)
			}
			if sampleSize < 1 {
				sampleSize = 1
			}
			parent := r.selectBest(r.randomSample(population, sampleSize))
			batch[i] = config.SearchSpace.Mutate(parent.arch)
			batch[i].Metadata.Generation = generation
		}
		for _, o := range evaluateBatch(ctx, config.NumWorkers, batch, func(c context.Context, a *searchspace.Architecture) (float64, error) {
			return r.evaluateArch(c, config, a)
		}) {
			if o.err != nil || o.arch == nil {
				continue
			}
			commit(o, generation)
			if len(population) >= config.PopulationSize {
				population = population[1:]
			}
			population = append(population, &individual{arch: o.arch, addedAt: tick, fitness: o.fitness})
			tick++
			if config.OnEvaluation != nil {
				pops := make([]*searchspace.Architecture, len(population))
				for i, ind := range population {
					pops[i] = ind.arch
				}
				config.OnEvaluation(EvaluationEvent{Architecture: o.arch, Fitness: o.fitness, EvaluationNumber: count, TotalEvaluations: config.MaxEvaluations, Duration: o.duration, BestSoFar: best, Generation: generation, Population: pops, StrategyRNG: r.rngSource.State, SearchSpaceRNG: config.SearchSpace.RNGState()})
			}
		}
		generation++
	}
	return finish(false)
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
