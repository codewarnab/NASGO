// Package utils provides utility functions for configuration, logging, etc.
package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"
)

// Config holds all configuration for the NAS system.
// This is the main configuration struct that gets loaded from YAML.
//
// Configuration philosophy:
// - All settings have sensible defaults
// - YAML file overrides defaults
// - Environment variables override YAML (for containerized deployments)
// - CLI flags override everything (for one-off experiments)
type Config struct {
	// Experiment identifies this run
	Experiment ExperimentConfig `yaml:"experiment"`

	// Search configures the search algorithm
	Search SearchConfig `yaml:"search"`

	// Evaluator configures how architectures are evaluated
	Evaluator EvaluatorCfg `yaml:"evaluator"`

	// Storage configures where results are saved
	Storage StorageConfig `yaml:"storage"`

	// Logging configures log output
	Logging LoggingConfig `yaml:"logging"`
}

// ExperimentConfig identifies and describes the experiment.
type ExperimentConfig struct {
	// Name is a human-readable experiment name
	Name string `yaml:"name"`

	// Description explains what this experiment tests
	Description string `yaml:"description,omitempty"`

	// Tags for organizing experiments (e.g., ["baseline", "cifar10"])
	Tags []string `yaml:"tags,omitempty"`

	// Seed for reproducibility (-1 for random)
	Seed int64 `yaml:"seed"`
}

// SearchConfig configures the search algorithm.
type SearchConfig struct {
	// Strategy is the search algorithm: "random", "evolutionary", "regularized"
	Strategy string `yaml:"strategy"`

	// MaxEvaluations is the total evaluation budget
	MaxEvaluations int `yaml:"max_evaluations"`

	// PopulationSize for evolutionary algorithms
	PopulationSize int `yaml:"population_size"`

	// TournamentSize for regularized evolution
	TournamentSize int `yaml:"tournament_size"`

	// NumWorkers for parallel evaluation
	NumWorkers int `yaml:"num_workers"`

	// SearchSpace configures the architecture search space
	SearchSpace SearchSpaceConfig `yaml:"search_space"`
}

// SearchSpaceConfig defines the search space.
type SearchSpaceConfig struct {
	// NumNodes is intermediate nodes per cell
	NumNodes int `yaml:"num_nodes"`

	// NumInputNodes is input nodes per cell (typically 2)
	NumInputNodes int `yaml:"num_input_nodes"`

	// EdgesPerNode is edges per node (typically 2)
	EdgesPerNode int `yaml:"edges_per_node"`

	// Operations is the list of allowed operations
	// Options: identity, conv_3x3, conv_5x5, sep_conv_3x3, etc.
	Operations []string `yaml:"operations"`
}

// EvaluatorCfg configures the evaluator.
type EvaluatorCfg struct {
	// Type is the evaluator type: "proxy", "trainer", "combined"
	Type string `yaml:"type"`

	// Dataset for training (cifar10, cifar100, imagenet)
	Dataset string `yaml:"dataset"`

	// DataPath is path to the dataset
	DataPath string `yaml:"data_path"`

	// Epochs for training-based evaluation
	Epochs int `yaml:"epochs"`

	// BatchSize for training
	BatchSize int `yaml:"batch_size"`

	// Channels is the initial channel count
	Channels int `yaml:"channels"`

	// Layers is total cells in the network
	Layers int `yaml:"layers"`

	// LearningRate for optimizer
	LearningRate float64 `yaml:"learning_rate"`

	// UseGPU enables GPU acceleration
	UseGPU bool `yaml:"use_gpu"`

	// GPUDevice specifies which GPU (e.g., "cuda:0")
	GPUDevice string `yaml:"gpu_device"`

	// PythonPath is path to Python executable
	PythonPath string `yaml:"python_path"`

	// ScriptPath is path to training script
	ScriptPath string `yaml:"script_path"`

	// Timeout for single evaluation
	Timeout time.Duration `yaml:"timeout"`

	// ProxyThreshold for combined evaluator
	ProxyThreshold float64 `yaml:"proxy_threshold"`
}

// StorageConfig configures result storage.
type StorageConfig struct {
	// Type is storage type: "sqlite", "file", "none"
	Type string `yaml:"type"`

	// Path is the storage path (DB file or directory)
	Path string `yaml:"path"`

	// SaveHistory saves all evaluated architectures (can be large)
	SaveHistory bool `yaml:"save_history"`

	// CheckpointInterval saves intermediate results every N evaluations
	CheckpointInterval int `yaml:"checkpoint_interval"`
}

// LoggingConfig configures logging.
type LoggingConfig struct {
	// Level is the log level: debug, info, warn, error
	Level string `yaml:"level"`

	// Format is the log format: json, text
	Format string `yaml:"format"`

	// File is optional log file path (empty = stdout)
	File string `yaml:"file,omitempty"`

	// IncludeSource includes source file/line in logs
	IncludeSource bool `yaml:"include_source"`
}

// DefaultConfig returns sensible defaults for quick experiments.
func DefaultConfig() *Config {
	return &Config{
		Experiment: ExperimentConfig{
			Name: "nas-experiment",
			Seed: 42,
		},
		Search: SearchConfig{
			Strategy:       "regularized",
			MaxEvaluations: 1000,
			PopulationSize: 100,
			TournamentSize: 25,
			NumWorkers:     1,
			SearchSpace: SearchSpaceConfig{
				NumNodes:      4,
				NumInputNodes: 2,
				EdgesPerNode:  2,
				Operations: []string{
					"identity", "conv_3x3", "conv_5x5",
					"sep_conv_3x3", "sep_conv_5x5",
					"max_pool_3x3", "avg_pool_3x3", "zero",
				},
			},
		},
		Evaluator: EvaluatorCfg{
			Type:         "proxy",
			Dataset:      "cifar10",
			DataPath:     "./data",
			Epochs:       50,
			BatchSize:    64,
			Channels:     16,
			Layers:       8,
			LearningRate: 0.025,
			UseGPU:       true,
			GPUDevice:    "cuda:0",
			PythonPath:   "python",
			ScriptPath:   "./scripts/train.py",
			Timeout:      30 * time.Minute,
		},
		Storage: StorageConfig{
			Type:               "sqlite",
			Path:               "./experiments.db",
			SaveHistory:        true,
			CheckpointInterval: 100,
		},
		Logging: LoggingConfig{
			Level:  "info",
			Format: "text",
		},
	}
}

// LoadConfig loads configuration from a YAML file.
// Missing fields use defaults.
func LoadConfig(path string) (*Config, error) {
	// Start with defaults
	cfg := DefaultConfig()

	// Read file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading config file: %w", err)
	}

	// Unmarshal into config (merges with defaults)
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("parsing config YAML: %w", err)
	}

	return cfg, nil
}

// Save writes configuration to a YAML file.
func (c *Config) Save(path string) error {
	data, err := yaml.Marshal(c)
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}

	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("creating config directory: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("writing config file: %w", err)
	}

	return nil
}

// ToJSON serializes the config to JSON bytes.
func (c *Config) ToJSON() ([]byte, error) {
	return json.Marshal(c)
}

// Validate checks that the configuration is valid.
func (c *Config) Validate() error {
	// Check strategy
	validStrategies := map[string]bool{
		"random": true, "evolutionary": true, "regularized": true,
	}
	if !validStrategies[c.Search.Strategy] {
		return fmt.Errorf("invalid search strategy: %s", c.Search.Strategy)
	}

	// Check evaluator type
	validEvaluators := map[string]bool{
		"proxy": true, "trainer": true, "combined": true,
	}
	if !validEvaluators[c.Evaluator.Type] {
		return fmt.Errorf("invalid evaluator type: %s", c.Evaluator.Type)
	}

	// Check numerical bounds
	if c.Search.MaxEvaluations < 1 {
		return fmt.Errorf("max_evaluations must be >= 1")
	}
	if c.Search.PopulationSize < 1 {
		return fmt.Errorf("population_size must be >= 1")
	}

	return nil
}
