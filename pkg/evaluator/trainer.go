package evaluator

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"nas-go/pkg/searchspace"
)

// TrainerEvaluator evaluates architectures by training them with PyTorch.
// This provides ground-truth accuracy but is computationally expensive.
//
// Architecture of the integration:
// 1. Go serializes architecture to JSON
// 2. Go calls Python training script via subprocess
// 3. Python builds the network, trains it, returns metrics as JSON
// 4. Go parses the results
//
// Why subprocess instead of embedded Python?
// - Simpler dependency management
// - Use any Python version/environment
// - GPU support just works
// - Can run on remote machines via SSH
type TrainerEvaluator struct {
	config     TrainerConfig
	scriptPath string
	pythonPath string
}

// TrainerConfig configures the training process.
type TrainerConfig struct {
	// EvaluatorConfig contains common settings
	EvaluatorConfig

	// Epochs to train (more = more accurate but slower)
	Epochs int `json:"epochs"`

	// LearningRate for optimizer
	LearningRate float64 `json:"learning_rate"`

	// WeightDecay for regularization
	WeightDecay float64 `json:"weight_decay"`

	// CutoutLength for cutout augmentation (0 to disable)
	CutoutLength int `json:"cutout_length"`

	// UseCosineAnnealing for learning rate schedule
	UseCosineAnnealing bool `json:"use_cosine_annealing"`

	// Timeout for single architecture evaluation
	Timeout time.Duration `json:"timeout"`

	// TempDir for temporary files
	TempDir string `json:"temp_dir"`
}

// DefaultTrainerConfig returns defaults for quick evaluation.
func DefaultTrainerConfig() TrainerConfig {
	return TrainerConfig{
		EvaluatorConfig:    DefaultEvaluatorConfig(),
		Epochs:             50, // Reduced epochs for faster search
		LearningRate:       0.025,
		WeightDecay:        3e-4,
		CutoutLength:       16,
		UseCosineAnnealing: true,
		Timeout:            30 * time.Minute,
		TempDir:            os.TempDir(),
	}
}

// NewTrainerEvaluator creates a trainer that uses Python for evaluation.
//
// Parameters:
//   - config: Training configuration
//   - scriptPath: Path to the Python training script (train.py)
//   - pythonPath: Path to Python executable (default: "python")
func NewTrainerEvaluator(config TrainerConfig, scriptPath, pythonPath string) (*TrainerEvaluator, error) {
	// Validate script exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("training script not found: %s", scriptPath)
	}

	if pythonPath == "" {
		pythonPath = "python"
	}

	// Verify Python is available
	cmd := exec.Command(pythonPath, "--version")
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("python not available at %s: %w", pythonPath, err)
	}

	return &TrainerEvaluator{
		config:     config,
		scriptPath: scriptPath,
		pythonPath: pythonPath,
	}, nil
}

// Name returns the evaluator name.
func (t *TrainerEvaluator) Name() string {
	return "TrainerEvaluator"
}

// EstimatedTime returns estimated training time.
func (t *TrainerEvaluator) EstimatedTime() time.Duration {
	// Rough estimate based on epochs
	return time.Duration(t.config.Epochs) * 30 * time.Second
}

// Evaluate trains the architecture and returns validation accuracy.
func (t *TrainerEvaluator) Evaluate(ctx context.Context, arch *searchspace.Architecture) (*EvaluationResult, error) {
	startTime := time.Now()

	// Create timeout context
	evalCtx, cancel := context.WithTimeout(ctx, t.config.Timeout)
	defer cancel()

	// Serialize architecture to temp file
	archPath, err := t.writeArchitecture(arch)
	if err != nil {
		return nil, fmt.Errorf("writing architecture: %w", err)
	}
	defer os.Remove(archPath)

	// Build command arguments
	args := t.buildArgs(archPath)

	// Execute Python training script
	cmd := exec.CommandContext(evalCtx, t.pythonPath, args...)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Run training
	if err := cmd.Run(); err != nil {
		// Check if it was a timeout
		if evalCtx.Err() == context.DeadlineExceeded {
			return &EvaluationResult{
				Fitness:        0,
				Error:          "training timeout exceeded",
				EvaluationTime: time.Since(startTime),
			}, nil
		}
		return nil, fmt.Errorf("training failed: %w\nstderr: %s", err, stderr.String())
	}

	// Parse results from stdout
	result, err := t.parseResults(stdout.Bytes())
	if err != nil {
		return nil, fmt.Errorf("parsing results: %w\nstdout: %s", err, stdout.String())
	}

	result.EvaluationTime = time.Since(startTime)
	return result, nil
}

// writeArchitecture serializes architecture to a temp JSON file.
func (t *TrainerEvaluator) writeArchitecture(arch *searchspace.Architecture) (string, error) {
	data, err := arch.ToJSON(true)
	if err != nil {
		return "", err
	}

	filename := filepath.Join(t.config.TempDir, fmt.Sprintf("arch_%s.json", arch.ID[:8]))
	if err := os.WriteFile(filename, data, 0644); err != nil {
		return "", err
	}

	return filename, nil
}

// buildArgs constructs command-line arguments for the training script.
func (t *TrainerEvaluator) buildArgs(archPath string) []string {
	return []string{
		t.scriptPath,
		"--arch", archPath,
		"--dataset", t.config.Dataset,
		"--data-path", t.config.DataPath,
		"--epochs", fmt.Sprintf("%d", t.config.Epochs),
		"--batch-size", fmt.Sprintf("%d", t.config.BatchSize),
		"--lr", fmt.Sprintf("%f", t.config.LearningRate),
		"--weight-decay", fmt.Sprintf("%f", t.config.WeightDecay),
		"--channels", fmt.Sprintf("%d", t.config.Channels),
		"--layers", fmt.Sprintf("%d", t.config.Layers),
		"--cutout-length", fmt.Sprintf("%d", t.config.CutoutLength),
		"--output-format", "json",
	}
}

// TrainingOutput is the JSON structure returned by the Python script.
type TrainingOutput struct {
	Accuracy           float64 `json:"accuracy"`
	ValidationAccuracy float64 `json:"validation_accuracy"`
	TrainingLoss       float64 `json:"training_loss"`
	Parameters         int64   `json:"parameters"`
	FLOPs              int64   `json:"flops"`
	Epochs             int     `json:"epochs"`
	Error              string  `json:"error,omitempty"`
}

// parseResults parses the JSON output from the training script.
func (t *TrainerEvaluator) parseResults(output []byte) (*EvaluationResult, error) {
	// Find JSON in output (script may print other stuff first)
	jsonStart := bytes.Index(output, []byte("{"))
	jsonEnd := bytes.LastIndex(output, []byte("}"))

	if jsonStart < 0 || jsonEnd < 0 || jsonEnd <= jsonStart {
		return nil, fmt.Errorf("no JSON found in output")
	}

	jsonBytes := output[jsonStart : jsonEnd+1]

	var trainOutput TrainingOutput
	if err := json.Unmarshal(jsonBytes, &trainOutput); err != nil {
		return nil, fmt.Errorf("parsing JSON: %w", err)
	}

	return &EvaluationResult{
		Fitness:            trainOutput.ValidationAccuracy,
		Accuracy:           trainOutput.Accuracy,
		ValidationAccuracy: trainOutput.ValidationAccuracy,
		TrainingLoss:       trainOutput.TrainingLoss,
		Parameters:         trainOutput.Parameters,
		FLOPs:              trainOutput.FLOPs,
		Epochs:             trainOutput.Epochs,
		Error:              trainOutput.Error,
	}, nil
}
