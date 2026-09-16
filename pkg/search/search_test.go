package search

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"nas-go/pkg/searchspace"
)

// ─── Helper Functions ───────────────────────────────────────────────────────

// simpleEvaluator returns a deterministic fitness based on parameter count.
// Used for testing search logic without actual neural network training.
func simpleEvaluator(ctx context.Context, arch *searchspace.Architecture) (float64, error) {
	// Use inverse parameter count as fitness (smaller = better)
	params := float64(arch.ParameterEstimate())
	if params == 0 {
		return 1.0, nil
	}
	return 1.0 / (1.0 + params), nil
}

// makeTestConfig creates a minimal search config for testing.
func makeTestConfig() SearchConfig {
	space := searchspace.DefaultSearchSpace()
	space.SetSeed(42)

	return SearchConfig{
		SearchSpace:    space,
		MaxEvaluations: 20,
		PopulationSize: 10,
		TournamentSize: 5,
		NumWorkers:     1,
		Seed:           42,
		EvaluatorFunc:  simpleEvaluator,
	}
}

// ─── RandomSearch Tests ─────────────────────────────────────────────────────

func TestRandomSearchName(t *testing.T) {
	rs := NewRandomSearch(42)
	if got := rs.Name(); got != "RandomSearch" {
		t.Errorf("Name() = %q, want %q", got, "RandomSearch")
	}
}

func TestRandomSearchBasic(t *testing.T) {
	rs := NewRandomSearch(42)
	ctx := context.Background()
	config := makeTestConfig()

	result, err := rs.Search(ctx, config)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if result.BestArchitecture == nil {
		t.Fatal("BestArchitecture should not be nil")
	}
	if result.BestFitness <= 0 {
		t.Errorf("BestFitness = %f, want > 0", result.BestFitness)
	}
	if result.TotalEvaluations != config.MaxEvaluations {
		t.Errorf("TotalEvaluations = %d, want %d", result.TotalEvaluations, config.MaxEvaluations)
	}
	if result.StrategyName != "RandomSearch" {
		t.Errorf("StrategyName = %q, want %q", result.StrategyName, "RandomSearch")
	}
	if result.SearchDuration <= 0 {
		t.Error("SearchDuration should be > 0")
	}
}

func TestRandomSearchReproducibility(t *testing.T) {
	config := makeTestConfig()
	ctx := context.Background()

	rs1 := NewRandomSearch(42)
	result1, _ := rs1.Search(ctx, config)

	rs2 := NewRandomSearch(42)
	result2, _ := rs2.Search(ctx, config)

	if result1.BestFitness != result2.BestFitness {
		t.Errorf("results differ: %f vs %f", result1.BestFitness, result2.BestFitness)
	}
}

func TestRandomSearchCancellation(t *testing.T) {
	rs := NewRandomSearch(42)
	config := makeTestConfig()
	config.MaxEvaluations = 10000 // large budget

	// Cancel after a short time
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	result, err := rs.Search(ctx, config)
	if err == nil {
		// It's possible the search completes before timeout (unlikely with 10000 evals)
		if !result.Cancelled && result.TotalEvaluations >= config.MaxEvaluations {
			return // Search completed naturally
		}
	}

	if result != nil && result.Cancelled {
		if result.TotalEvaluations >= config.MaxEvaluations {
			t.Error("cancelled search should have fewer evaluations than budget")
		}
	}
}

func TestRandomSearchNoEvaluator(t *testing.T) {
	rs := NewRandomSearch(42)
	ctx := context.Background()
	config := makeTestConfig()
	config.EvaluatorFunc = nil // No evaluator - should use parameter count

	result, err := rs.Search(ctx, config)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if result.BestArchitecture == nil {
		t.Fatal("should find a best architecture even without evaluator")
	}
}

func TestRandomSearchWithCallback(t *testing.T) {
	rs := NewRandomSearch(42)
	ctx := context.Background()
	config := makeTestConfig()

	callbackCount := 0
	config.OnEvaluation = func(event EvaluationEvent) {
		callbackCount++
		if event.EvaluationNumber < 1 {
			t.Error("EvaluationNumber should be >= 1")
		}
	}

	_, err := rs.Search(ctx, config)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if callbackCount != config.MaxEvaluations {
		t.Errorf("callback called %d times, want %d", callbackCount, config.MaxEvaluations)
	}
}

// ─── EvolutionarySearch Tests ───────────────────────────────────────────────

func TestEvolutionarySearchName(t *testing.T) {
	es := NewEvolutionarySearch(42)
	if got := es.Name(); got != "EvolutionarySearch" {
		t.Errorf("Name() = %q, want %q", got, "EvolutionarySearch")
	}
}

func TestEvolutionarySearchBasic(t *testing.T) {
	es := NewEvolutionarySearch(42)
	ctx := context.Background()
	config := makeTestConfig()

	result, err := es.Search(ctx, config)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if result.BestArchitecture == nil {
		t.Fatal("BestArchitecture should not be nil")
	}
	if result.BestFitness <= 0 {
		t.Errorf("BestFitness = %f, want > 0", result.BestFitness)
	}
	if result.StrategyName != "EvolutionarySearch" {
		t.Errorf("StrategyName = %q, want %q", result.StrategyName, "EvolutionarySearch")
	}
}

func TestEvolutionarySearchEvolution(t *testing.T) {
	es := NewEvolutionarySearch(42)
	ctx := context.Background()
	config := makeTestConfig()
	config.MaxEvaluations = 30 // Enough for some evolution

	result, err := es.Search(ctx, config)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if result.FinalGeneration < 1 {
		t.Errorf("FinalGeneration = %d, want >= 1", result.FinalGeneration)
	}
}

// ─── RegularizedEvolution Tests ─────────────────────────────────────────────

func TestRegularizedEvolutionName(t *testing.T) {
	re := NewRegularizedEvolution(42)
	if got := re.Name(); got != "RegularizedEvolution" {
		t.Errorf("Name() = %q, want %q", got, "RegularizedEvolution")
	}
}

func TestRegularizedEvolutionBasic(t *testing.T) {
	re := NewRegularizedEvolution(42)
	ctx := context.Background()
	config := makeTestConfig()

	result, err := re.Search(ctx, config)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	if result.BestArchitecture == nil {
		t.Fatal("BestArchitecture should not be nil")
	}
	if result.BestFitness <= 0 {
		t.Errorf("BestFitness = %f, want > 0", result.BestFitness)
	}
	if result.StrategyName != "RegularizedEvolution" {
		t.Errorf("StrategyName = %q, want %q", result.StrategyName, "RegularizedEvolution")
	}
}

func TestRegularizedEvolutionAging(t *testing.T) {
	re := NewRegularizedEvolution(42)
	ctx := context.Background()
	config := makeTestConfig()
	config.MaxEvaluations = 30 // Enough for aging to kick in
	config.PopulationSize = 10

	result, err := re.Search(ctx, config)
	if err != nil {
		t.Fatalf("Search() error: %v", err)
	}

	// Should have multiple generations
	if result.FinalGeneration < 1 {
		t.Errorf("FinalGeneration = %d, want >= 1", result.FinalGeneration)
	}
	if result.TotalEvaluations < config.MaxEvaluations {
		t.Errorf("TotalEvaluations = %d, want %d", result.TotalEvaluations, config.MaxEvaluations)
	}
}

func TestRegularizedEvolutionCancellation(t *testing.T) {
	re := NewRegularizedEvolution(42)
	config := makeTestConfig()
	config.MaxEvaluations = 10000

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	result, _ := re.Search(ctx, config)

	if result != nil && result.Cancelled {
		if result.TotalEvaluations >= config.MaxEvaluations {
			t.Error("cancelled search should have fewer evaluations than budget")
		}
	}
}

// ─── SearchResult Tests ─────────────────────────────────────────────────────

func TestSearchResultSummary(t *testing.T) {
	normal := searchspace.NewCell(searchspace.NormalCell, 4, 2, 2)
	reduction := searchspace.NewCell(searchspace.ReductionCell, 4, 2, 2)
	arch := searchspace.NewArchitecture(normal, reduction)

	result := &SearchResult{
		BestArchitecture: arch,
		BestFitness:      0.95,
		TotalEvaluations: 100,
		SearchDuration:   5 * time.Second,
		StrategyName:     "test",
		Cancelled:        false,
	}

	summary := result.Summary()
	if summary == "" {
		t.Error("Summary() should not be empty")
	}
}

func TestDefaultSearchConfig(t *testing.T) {
	space := searchspace.DefaultSearchSpace()
	config := DefaultSearchConfig(space)

	if config.SearchSpace != space {
		t.Error("SearchSpace should be the one passed in")
	}
	if config.MaxEvaluations < 1 {
		t.Error("MaxEvaluations should be >= 1")
	}
	if config.PopulationSize < 1 {
		t.Error("PopulationSize should be >= 1")
	}
}

// ─── Comparison Test ────────────────────────────────────────────────────────

func TestAllStrategiesCompile(t *testing.T) {
	// Verify all strategies implement Searcher
	strategies := []Searcher{
		NewRandomSearch(42),
		NewEvolutionarySearch(42),
		NewRegularizedEvolution(42),
	}

	for _, s := range strategies {
		if s.Name() == "" {
			t.Error("strategy name should not be empty")
		}
	}
}

func TestRandomSearchUsesWorkersWithoutExceedingBudget(t *testing.T) {
	space := searchspace.DefaultSearchSpace()
	var current, peak int32
	cfg := DefaultSearchConfig(space)
	cfg.MaxEvaluations = 7
	cfg.NumWorkers = 3
	cfg.EvaluatorFunc = func(ctx context.Context, a *searchspace.Architecture) (float64, error) {
		n := atomic.AddInt32(&current, 1)
		for {
			p := atomic.LoadInt32(&peak)
			if n <= p || atomic.CompareAndSwapInt32(&peak, p, n) {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
		atomic.AddInt32(&current, -1)
		return 1, nil
	}
	result, err := NewRandomSearch(42).Search(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if result.TotalEvaluations != 7 {
		t.Fatalf("evaluations=%d", result.TotalEvaluations)
	}
	if peak < 2 {
		t.Fatalf("peak concurrency=%d", peak)
	}
}

func TestEvolutionaryStrategiesUseWorkers(t *testing.T) {
	for _, tc := range []struct {
		name string
		make func() Searcher
	}{
		{name: "evolutionary", make: func() Searcher { return NewEvolutionarySearch(42) }},
		{name: "regularized", make: func() Searcher { return NewRegularizedEvolution(42) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := makeTestConfig()
			cfg.NumWorkers = 3
			cfg.PopulationSize = 6
			cfg.MaxEvaluations = 12
			var active, peak int32
			cfg.EvaluatorFunc = func(context.Context, *searchspace.Architecture) (float64, error) {
				n := atomic.AddInt32(&active, 1)
				for {
					old := atomic.LoadInt32(&peak)
					if n <= old || atomic.CompareAndSwapInt32(&peak, old, n) {
						break
					}
				}
				time.Sleep(5 * time.Millisecond)
				atomic.AddInt32(&active, -1)
				return 1, nil
			}
			result, err := tc.make().Search(context.Background(), cfg)
			if err != nil {
				t.Fatal(err)
			}
			if peak < 2 {
				t.Fatalf("peak concurrency=%d", peak)
			}
			if result.TotalEvaluations != cfg.MaxEvaluations {
				t.Fatalf("evaluations=%d, want %d", result.TotalEvaluations, cfg.MaxEvaluations)
			}
		})
	}
}

func TestEvolutionaryStrategiesCancelMidDispatchWithoutPanic(t *testing.T) {
	for _, tc := range []struct {
		name string
		make func() Searcher
	}{
		{name: "evolutionary", make: func() Searcher { return NewEvolutionarySearch(42) }},
		{name: "regularized", make: func() Searcher { return NewRegularizedEvolution(42) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := makeTestConfig()
			cfg.NumWorkers = 8
			cfg.PopulationSize = 8
			cfg.MaxEvaluations = 100
			started := make(chan struct{}, 1)
			cfg.EvaluatorFunc = func(ctx context.Context, _ *searchspace.Architecture) (float64, error) {
				select {
				case started <- struct{}{}:
				default:
				}
				<-ctx.Done()
				return 0, ctx.Err()
			}
			ctx, cancel := context.WithCancel(context.Background())
			go func() { <-started; cancel() }()
			result, err := tc.make().Search(ctx, cfg)
			if err == nil || result == nil || !result.Cancelled {
				t.Fatalf("result=%+v err=%v", result, err)
			}
		})
	}
}

func TestEvolutionaryInitializationPersistentErrorsAreBounded(t *testing.T) {
	cfg := makeTestConfig()
	cfg.PopulationSize = 4
	cfg.NumWorkers = 2
	cfg.MaxEvaluations = 20
	var calls int32
	cfg.EvaluatorFunc = func(context.Context, *searchspace.Architecture) (float64, error) {
		atomic.AddInt32(&calls, 1)
		return 0, context.DeadlineExceeded
	}
	result, err := NewEvolutionarySearch(42).Search(context.Background(), cfg)
	if err == nil || result != nil {
		t.Fatalf("result=%+v err=%v", result, err)
	}
	if calls > 12 {
		t.Fatalf("unbounded evaluator calls: %d", calls)
	}
}
