// Package evaluator provides methods to evaluate neural architecture quality.
// Evaluation is the most computationally expensive part of NAS, so we provide
// multiple strategies with different speed/accuracy tradeoffs:
//
// - ProxyEvaluator: Fast heuristics (seconds per architecture)
// - TrainerEvaluator: Full training via Python (minutes to hours)
//
// The Evaluator interface allows swapping strategies based on needs.
package evaluator

import (
	"context"
	"fmt"
	"time"

	"nas-go/pkg/searchspace"
)

// Evaluator is the interface for architecture evaluation strategies.
// All evaluators return a fitness score where HIGHER IS BETTER.
//
// Why an interface?
// 1. Different evaluation methods for different use cases
// 2. Easy to mock for testing search algorithms
// 3. Can combine evaluators (e.g., proxy filter + full training)
type Evaluator interface {
	// Evaluate computes the fitness of an architecture.
	// Higher fitness = better architecture.
	//
	// Parameters:
	//   - ctx: Context for cancellation and timeout
	//   - arch: The architecture to evaluate
	//
	// Returns:
	//   - result: Detailed evaluation result
	//   - error: If evaluation fails
	Evaluate(ctx context.Context, arch *searchspace.Architecture) (*EvaluationResult, error)

	// Name returns the evaluator name for logging.
	Name() string

	// EstimatedTime returns the estimated time to evaluate one architecture.
	// Used for progress estimation and timeout configuration.
	EstimatedTime() time.Duration
}

// EvaluationResult contains detailed results from evaluating an architecture.
// More detailed than just fitness - useful for analysis and debugging.
type EvaluationResult struct {
	// Fitness is the primary score (higher is better).
	// This is what the search algorithm optimizes.
	Fitness float64 `json:"fitness"`

	// Accuracy is the classification accuracy (0.0 to 1.0).
	// Only set for training-based evaluation.
	Accuracy float64 `json:"accuracy,omitempty"`

	// ValidationAccuracy is accuracy on validation set.
	ValidationAccuracy float64 `json:"validation_accuracy,omitempty"`

	// Parameters is the total parameter count of the architecture.
	Parameters int64 `json:"parameters"`

	// FLOPs is the floating point operations per forward pass.
	// Useful for efficiency-aware NAS.
	FLOPs int64 `json:"flops,omitempty"`

	// Latency is the inference time on target hardware.
	LatencyMs float64 `json:"latency_ms,omitempty"`

	// ProxyScores contains individual proxy metric scores.
	// Useful for debugging and analysis.
	ProxyScores map[string]float64 `json:"proxy_scores,omitempty"`

	// TrainingLoss is the final training loss.
	TrainingLoss float64 `json:"training_loss,omitempty"`

	// Epochs is how many epochs were trained.
	Epochs int `json:"epochs,omitempty"`

	// EvaluationTime is how long evaluation took.
	EvaluationTime time.Duration `json:"evaluation_time"`

	// Error contains any error message (evaluation can partially succeed).
	Error string `json:"error,omitempty"`
}

// ToFitness converts the result to a simple fitness value for search.
// This is the value that search algorithms optimize.
func (r *EvaluationResult) ToFitness() float64 {
	return r.Fitness
}

// Summary returns a human-readable summary.
func (r *EvaluationResult) Summary() string {
	return fmt.Sprintf(
		"Fitness: %.4f, Params: %d, Time: %s",
		r.Fitness, r.Parameters, r.EvaluationTime.Round(time.Millisecond),
	)
}

// EvaluatorConfig holds common configuration for evaluators.
type EvaluatorConfig struct {
	// Dataset specifies which dataset to use for evaluation.
	// Options: "cifar10", "cifar100", "imagenet"
	Dataset string `json:"dataset"`

	// DataPath is the path to the dataset.
	DataPath string `json:"data_path"`

	// NumClasses is the number of output classes.
	NumClasses int `json:"num_classes"`

	// Channels is the initial channel count.
	// The network scales up channels through reduction cells.
	Channels int `json:"channels"`

	// Layers is the total number of cells in the network.
	Layers int `json:"layers"`

	// BatchSize for training/evaluation.
	BatchSize int `json:"batch_size"`

	// UseGPU enables GPU acceleration.
	UseGPU bool `json:"use_gpu"`

	// GPUDevice specifies which GPU to use (e.g., "cuda:0").
	GPUDevice string `json:"gpu_device"`
}

// DefaultEvaluatorConfig returns sensible defaults for CIFAR-10.
func DefaultEvaluatorConfig() EvaluatorConfig {
	return EvaluatorConfig{
		Dataset:    "cifar10",
		DataPath:   "./data",
		NumClasses: 10,
		Channels:   16,
		Layers:     8,
		BatchSize:  64,
		UseGPU:     true,
		GPUDevice:  "cuda:0",
	}
}

// CombinedEvaluator chains multiple evaluators.
// Useful for filtering with proxy metrics before expensive training.
type CombinedEvaluator struct {
	// Evaluators are applied in order
	evaluators []Evaluator
	// Thresholds: if an architecture scores below threshold[i],
	// skip remaining evaluators
	thresholds []float64
}

// NewCombinedEvaluator creates an evaluator that applies multiple evaluators in sequence.
// If an architecture fails to meet a threshold, subsequent evaluators are skipped.
//
// Example:
//
//	combined := NewCombinedEvaluator(
//	    []Evaluator{proxyEval, trainerEval},
//	    []float64{0.5, 0},  // Must get 0.5+ from proxy to proceed to training
//	)
func NewCombinedEvaluator(evaluators []Evaluator, thresholds []float64) *CombinedEvaluator {
	return &CombinedEvaluator{
		evaluators: evaluators,
		thresholds: thresholds,
	}
}

// Evaluate runs evaluators in sequence, stopping early if thresholds not met.
func (c *CombinedEvaluator) Evaluate(ctx context.Context, arch *searchspace.Architecture) (*EvaluationResult, error) {
	var lastResult *EvaluationResult

	for i, eval := range c.evaluators {
		select {
		case <-ctx.Done():
			return lastResult, ctx.Err()
		default:
		}

		result, err := eval.Evaluate(ctx, arch)
		if err != nil {
			return result, fmt.Errorf("%s: %w", eval.Name(), err)
		}

		lastResult = result

		// Check threshold (if provided)
		if i < len(c.thresholds) && result.Fitness < c.thresholds[i] {
			result.Error = fmt.Sprintf("below threshold %.2f at stage %s", c.thresholds[i], eval.Name())
			return result, nil
		}
	}

	return lastResult, nil
}

func (c *CombinedEvaluator) Name() string {
	return "CombinedEvaluator"
}

func (c *CombinedEvaluator) EstimatedTime() time.Duration {
	var total time.Duration
	for _, e := range c.evaluators {
		total += e.EstimatedTime()
	}
	return total
}
