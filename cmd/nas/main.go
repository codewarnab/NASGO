// Package main is the CLI entry point for the NAS system.
// It provides a command-line interface for running neural architecture search
// with various search strategies and evaluation methods.
//
// Usage:
//
//	nas search --config configs/default.yaml
//	nas search --strategy random --evaluations 500
//	nas info --config configs/default.yaml
//	nas version
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"nas-go/pkg/evaluator"
	"nas-go/pkg/search"
	"nas-go/pkg/searchspace"
	"nas-go/pkg/storage"
	"nas-go/pkg/utils"

	"github.com/google/uuid"
)

// Build-time variables set by -ldflags.
var (
	version   = "dev"
	commit    = "none"
	buildDate = "unknown"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "search":
		if err := runSearch(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	case "info":
		if err := runInfo(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	case "version":
		fmt.Printf("nas %s (commit: %s, built: %s)\n", version, commit, buildDate)
		fmt.Printf("Go: %s, OS/Arch: %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

// printUsage prints the CLI help message.
func printUsage() {
	fmt.Println(`NAS - Neural Architecture Search in Go

Usage:
  nas <command> [flags]

Commands:
  search    Run architecture search
  info      Show search space information
  version   Print version information
  help      Show this help message

Use "nas <command> -h" for more information about a command.`)
}

// runSearch executes the architecture search.
func runSearch(args []string) error {
	// Parse flags
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	configPath := fs.String("config", "", "Path to YAML config file")
	strategy := fs.String("strategy", "", "Search strategy: random, evolutionary, regularized")
	maxEvals := fs.Int("evaluations", 0, "Maximum number of evaluations")
	popSize := fs.Int("population", 0, "Population size (evolutionary)")
	tournamentSize := fs.Int("tournament", 0, "Tournament size (regularized)")
	workers := fs.Int("workers", 0, "Number of parallel workers")
	seed := fs.Int64("seed", -2, "Random seed (-1 for random)")
	evalType := fs.String("evaluator", "", "Evaluator type: proxy, trainer, combined")
	dbPath := fs.String("db", "", "SQLite database path")
	logLevel := fs.String("log-level", "", "Log level: debug, info, warn, error")
	outputJSON := fs.Bool("json", false, "Output results as JSON")

	if err := fs.Parse(args); err != nil {
		return err
	}

	// Load configuration (defaults + file + flags)
	cfg := utils.DefaultConfig()
	if *configPath != "" {
		loaded, err := utils.LoadConfig(*configPath)
		if err != nil {
			return fmt.Errorf("loading config: %w", err)
		}
		cfg = loaded
	}

	// Override with CLI flags
	if *strategy != "" {
		cfg.Search.Strategy = *strategy
	}
	if *maxEvals > 0 {
		cfg.Search.MaxEvaluations = *maxEvals
	}
	if *popSize > 0 {
		cfg.Search.PopulationSize = *popSize
	}
	if *tournamentSize > 0 {
		cfg.Search.TournamentSize = *tournamentSize
	}
	if *workers > 0 {
		cfg.Search.NumWorkers = *workers
	}
	if *seed != -2 {
		cfg.Experiment.Seed = *seed
	}
	if *evalType != "" {
		cfg.Evaluator.Type = *evalType
	}
	if *dbPath != "" {
		cfg.Storage.Path = *dbPath
	}
	if *logLevel != "" {
		cfg.Logging.Level = *logLevel
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	// Initialize logger
	logger, err := utils.NewLogger(cfg.Logging)
	if err != nil {
		return fmt.Errorf("initializing logger: %w", err)
	}
	logger.Info("starting NAS search",
		"strategy", cfg.Search.Strategy,
		"max_evaluations", cfg.Search.MaxEvaluations,
		"seed", cfg.Experiment.Seed,
	)

	// Build search space from config
	space, err := buildSearchSpace(cfg)
	if err != nil {
		return fmt.Errorf("building search space: %w", err)
	}
	logger.Info("search space configured", "size", fmt.Sprintf("%.2e", space.Size()))

	// Build evaluator
	eval, err := buildEvaluator(cfg, logger)
	if err != nil {
		return fmt.Errorf("building evaluator: %w", err)
	}
	logger.Info("evaluator configured", "type", cfg.Evaluator.Type)

	// Initialize storage
	var store *storage.SQLiteStorage
	if cfg.Storage.Type == "sqlite" {
		store, err = storage.NewSQLiteStorage(cfg.Storage.Path)
		if err != nil {
			return fmt.Errorf("initializing storage: %w", err)
		}
		defer store.Close()
		logger.Info("storage initialized", "path", cfg.Storage.Path)
	}

	// Create experiment record
	experimentID := uuid.New().String()
	if store != nil {
		configJSON, _ := cfg.ToJSON()
		exp := storage.Experiment{
			ID:          experimentID,
			Name:        cfg.Experiment.Name,
			Description: cfg.Experiment.Description,
			ConfigJSON:  string(configJSON),
			Strategy:    cfg.Search.Strategy,
			StartedAt:   time.Now(),
		}
		if err := store.CreateExperiment(context.Background(), exp); err != nil {
			logger.Warn("failed to save experiment", "error", err)
		}
	}

	// Build search strategy
	searcher, err := buildSearcher(cfg)
	if err != nil {
		return fmt.Errorf("building searcher: %w", err)
	}

	// Configure search
	searchCfg := search.SearchConfig{
		SearchSpace:    space,
		MaxEvaluations: cfg.Search.MaxEvaluations,
		PopulationSize: cfg.Search.PopulationSize,
		TournamentSize: cfg.Search.TournamentSize,
		NumWorkers:     cfg.Search.NumWorkers,
		Seed:           cfg.Experiment.Seed,
		EvaluatorFunc: func(ctx context.Context, arch *searchspace.Architecture) (float64, error) {
			result, err := eval.Evaluate(ctx, arch)
			if err != nil {
				return 0, err
			}
			return result.Fitness, nil
		},
		OnEvaluation: func(event search.EvaluationEvent) {
			// Log progress
			if event.EvaluationNumber%100 == 0 || event.EvaluationNumber == 1 {
				logger.Progress(event.EvaluationNumber, event.TotalEvaluations, event.BestSoFar)
			}
			// Save to storage
			if store != nil && cfg.Storage.SaveHistory {
				if err := store.SaveArchitecture(context.Background(), experimentID, event.Architecture); err != nil {
					logger.Warn("failed to save architecture", "error", err)
				}
			}
		},
	}

	// Setup graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		logger.Info("received signal, shutting down gracefully", "signal", sig)
		cancel()
	}()

	// Run search
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  NAS Search: %s\n", searcher.Name())
	fmt.Printf("  Evaluations: %d | Strategy: %s\n", cfg.Search.MaxEvaluations, cfg.Search.Strategy)
	fmt.Printf("  Search Space: %.2e architectures\n", space.Size())
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println()

	searchResult, err := searcher.Search(ctx, searchCfg)
	if err != nil && !searchResult.Cancelled {
		return fmt.Errorf("search failed: %w", err)
	}

	// Print results
	fmt.Println()
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  Search Results")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println(searchResult.Summary())
	fmt.Println()

	if searchResult.BestArchitecture != nil {
		fmt.Println("Best Architecture:")
		fmt.Println(searchResult.BestArchitecture.Summary())
	}

	// JSON output if requested
	if *outputJSON && searchResult.BestArchitecture != nil {
		jsonData, err := searchResult.BestArchitecture.ToJSON(true)
		if err == nil {
			fmt.Println("\nArchitecture JSON:")
			fmt.Println(string(jsonData))
		}
	}

	// Update experiment in storage
	if store != nil {
		now := time.Now()
		exp := storage.Experiment{
			ID:               experimentID,
			Status:           "completed",
			CompletedAt:      &now,
			BestFitness:      &searchResult.BestFitness,
			TotalEvaluations: searchResult.TotalEvaluations,
		}
		if searchResult.BestArchitecture != nil {
			exp.BestArchID = &searchResult.BestArchitecture.ID
		}
		if searchResult.Cancelled {
			exp.Status = "cancelled"
		}
		if err := store.UpdateExperiment(context.Background(), exp); err != nil {
			logger.Warn("failed to update experiment", "error", err)
		}
	}

	logger.Info("search complete",
		"best_fitness", searchResult.BestFitness,
		"evaluations", searchResult.TotalEvaluations,
		"duration", searchResult.SearchDuration,
	)

	return nil
}

// runInfo prints search space information.
func runInfo(args []string) error {
	fs := flag.NewFlagSet("info", flag.ExitOnError)
	configPath := fs.String("config", "", "Path to YAML config file")
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg := utils.DefaultConfig()
	if *configPath != "" {
		loaded, err := utils.LoadConfig(*configPath)
		if err != nil {
			return fmt.Errorf("loading config: %w", err)
		}
		cfg = loaded
	}

	space, err := buildSearchSpace(cfg)
	if err != nil {
		return fmt.Errorf("building search space: %w", err)
	}

	fmt.Println("Search Space Information:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  Nodes per cell:     %d\n", cfg.Search.SearchSpace.NumNodes)
	fmt.Printf("  Input nodes:        %d\n", cfg.Search.SearchSpace.NumInputNodes)
	fmt.Printf("  Edges per node:     %d\n", cfg.Search.SearchSpace.EdgesPerNode)
	fmt.Printf("  Operations:         %s\n", strings.Join(cfg.Search.SearchSpace.Operations, ", "))
	fmt.Printf("  Total architectures: %.2e\n", space.Size())
	fmt.Println()
	fmt.Println("  Sample random architecture:")

	arch := space.SampleRandomArchitecture()
	fmt.Println(arch.Summary())

	return nil
}

// buildSearchSpace creates a SearchSpace from config.
func buildSearchSpace(cfg *utils.Config) (*searchspace.SearchSpace, error) {
	// Parse operations from string names
	ops := make([]searchspace.OperationType, 0, len(cfg.Search.SearchSpace.Operations))
	for _, name := range cfg.Search.SearchSpace.Operations {
		op, err := searchspace.ParseOperationType(name)
		if err != nil {
			return nil, fmt.Errorf("unknown operation %q: %w", name, err)
		}
		ops = append(ops, op)
	}

	return searchspace.NewSearchSpace(
		ops,
		cfg.Search.SearchSpace.NumNodes,
		cfg.Search.SearchSpace.NumInputNodes,
		cfg.Search.SearchSpace.EdgesPerNode,
		cfg.Experiment.Seed,
	)
}

// buildEvaluator creates an Evaluator from config.
func buildEvaluator(cfg *utils.Config, logger *utils.Logger) (evaluator.Evaluator, error) {
	switch cfg.Evaluator.Type {
	case "proxy":
		proxyConfig := evaluator.DefaultProxyConfig()
		return evaluator.NewProxyEvaluator(proxyConfig), nil

	case "trainer":
		trainerConfig := evaluator.TrainerConfig{
			EvaluatorConfig: evaluator.EvaluatorConfig{
				Dataset:    cfg.Evaluator.Dataset,
				DataPath:   cfg.Evaluator.DataPath,
				NumClasses: 10,
				Channels:   cfg.Evaluator.Channels,
				Layers:     cfg.Evaluator.Layers,
				BatchSize:  cfg.Evaluator.BatchSize,
				UseGPU:     cfg.Evaluator.UseGPU,
				GPUDevice:  cfg.Evaluator.GPUDevice,
			},
			Epochs:             cfg.Evaluator.Epochs,
			LearningRate:       cfg.Evaluator.LearningRate,
			Timeout:            cfg.Evaluator.Timeout,
		}
		return evaluator.NewTrainerEvaluator(trainerConfig, cfg.Evaluator.ScriptPath, cfg.Evaluator.PythonPath)

	case "combined":
		proxy := evaluator.NewProxyEvaluator(evaluator.DefaultProxyConfig())
		trainerCfg := evaluator.TrainerConfig{
			EvaluatorConfig: evaluator.DefaultEvaluatorConfig(),
			Epochs:          cfg.Evaluator.Epochs,
			LearningRate:    cfg.Evaluator.LearningRate,
			Timeout:         cfg.Evaluator.Timeout,
		}
		trainer, err := evaluator.NewTrainerEvaluator(trainerCfg, cfg.Evaluator.ScriptPath, cfg.Evaluator.PythonPath)
		if err != nil {
			logger.Warn("trainer not available, using proxy only", "error", err)
			return proxy, nil
		}
		return evaluator.NewCombinedEvaluator(
			[]evaluator.Evaluator{proxy, trainer},
			[]float64{cfg.Evaluator.ProxyThreshold, 0},
		), nil

	default:
		return nil, fmt.Errorf("unknown evaluator type: %s", cfg.Evaluator.Type)
	}
}

// buildSearcher creates a Searcher from config.
func buildSearcher(cfg *utils.Config) (search.Searcher, error) {
	switch cfg.Search.Strategy {
	case "random":
		return search.NewRandomSearch(cfg.Experiment.Seed), nil
	case "evolutionary":
		return search.NewEvolutionarySearch(cfg.Experiment.Seed), nil
	case "regularized":
		return search.NewRegularizedEvolution(cfg.Experiment.Seed), nil
	default:
		return nil, fmt.Errorf("unknown search strategy: %s", cfg.Search.Strategy)
	}
}
