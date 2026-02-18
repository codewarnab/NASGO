package evaluator

import (
	"context"
	"math"
	"time"

	"nas-go/pkg/searchspace"
)

// ProxyEvaluator uses fast heuristics to estimate architecture quality.
// These "zero-cost" metrics don't require any training, making them
// orders of magnitude faster than full evaluation.
//
// Typical use cases:
// 1. Filter obviously bad architectures before expensive training
// 2. Quick exploration of the search space
// 3. Warm starting evolutionary algorithms
//
// Proxy metrics implemented:
// - Parameter count (fewer = more efficient)
// - Operation diversity (more diverse = more expressive)
// - Connection density (moderate density preferred)
// - Skip ratio (some skips good, too many bad)
//
// Note: These are simplified proxies. Production systems would use
// gradient-based metrics like SynFlow, jacob_cov, or NASWOT.
//
// Reference: "Zero-Cost Proxies for Lightweight NAS"
// https://arxiv.org/abs/2101.08134
type ProxyEvaluator struct {
	config ProxyConfig
}

// ProxyConfig configures how proxy metrics are computed and weighted.
type ProxyConfig struct {
	// Weights for combining proxy scores
	// All weights should sum to 1.0
	ParamWeight     float64 `json:"param_weight"`
	DiversityWeight float64 `json:"diversity_weight"`
	DensityWeight   float64 `json:"density_weight"`
	SkipRatioWeight float64 `json:"skip_ratio_weight"`

	// Target values for normalization
	TargetParams    int     `json:"target_params"`     // "ideal" parameter count
	TargetSkipRatio float64 `json:"target_skip_ratio"` // ideal skip connection ratio
	TargetDensity   float64 `json:"target_density"`    // ideal connection density
}

// DefaultProxyConfig returns sensible defaults.
func DefaultProxyConfig() ProxyConfig {
	return ProxyConfig{
		ParamWeight:     0.3,
		DiversityWeight: 0.3,
		DensityWeight:   0.2,
		SkipRatioWeight: 0.2,

		TargetParams:    100, // prefer ~100 relative params
		TargetSkipRatio: 0.2, // 20% skip connections is good
		TargetDensity:   0.7, // 70% edges active
	}
}

// NewProxyEvaluator creates a proxy evaluator with the given config.
func NewProxyEvaluator(config ProxyConfig) *ProxyEvaluator {
	return &ProxyEvaluator{config: config}
}

// Name returns the evaluator name.
func (p *ProxyEvaluator) Name() string {
	return "ProxyEvaluator"
}

// EstimatedTime returns how long evaluation takes (very fast).
func (p *ProxyEvaluator) EstimatedTime() time.Duration {
	return 1 * time.Millisecond // Almost instant
}

// Evaluate computes proxy fitness for an architecture.
// Returns a score in [0, 1] where higher is better.
func (p *ProxyEvaluator) Evaluate(ctx context.Context, arch *searchspace.Architecture) (*EvaluationResult, error) {
	startTime := time.Now()

	// Check for cancellation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	result := &EvaluationResult{
		ProxyScores: make(map[string]float64),
	}

	// Compute individual proxy metrics
	paramScore := p.computeParamScore(arch)
	diversityScore := p.computeDiversityScore(arch)
	densityScore := p.computeDensityScore(arch)
	skipScore := p.computeSkipRatioScore(arch)

	// Store individual scores for debugging
	result.ProxyScores["params"] = paramScore
	result.ProxyScores["diversity"] = diversityScore
	result.ProxyScores["density"] = densityScore
	result.ProxyScores["skip_ratio"] = skipScore

	// Weighted combination
	fitness := p.config.ParamWeight*paramScore +
		p.config.DiversityWeight*diversityScore +
		p.config.DensityWeight*densityScore +
		p.config.SkipRatioWeight*skipScore

	result.Fitness = fitness
	result.Parameters = int64(arch.ParameterEstimate())
	result.EvaluationTime = time.Since(startTime)

	return result, nil
}

// computeParamScore scores based on parameter count.
// Prefers architectures close to target param count.
// Score in [0, 1] with 1 being exactly at target.
func (p *ProxyEvaluator) computeParamScore(arch *searchspace.Architecture) float64 {
	params := float64(arch.ParameterEstimate())
	target := float64(p.config.TargetParams)

	if target <= 0 {
		target = 100
	}

	// Use Gaussian-like scoring centered at target
	// Score = exp(-((params - target) / target)^2)
	ratio := (params - target) / target
	score := math.Exp(-ratio * ratio)

	return score
}

// computeDiversityScore measures operation diversity.
// Higher diversity = more types of operations used = more expressive.
func (p *ProxyEvaluator) computeDiversityScore(arch *searchspace.Architecture) float64 {
	// Count unique operations across both cells
	normalOps := arch.NormalCell.UsedOperations()
	reductionOps := arch.ReductionCell.UsedOperations()

	// Combine into unique set
	seen := make(map[searchspace.OperationType]bool)
	for _, op := range normalOps {
		seen[op] = true
	}
	for _, op := range reductionOps {
		seen[op] = true
	}

	uniqueOps := float64(len(seen))

	// Normalize by max possible operations
	// Assume ~9 different operation types as maximum
	maxOps := 9.0
	score := uniqueOps / maxOps

	if score > 1.0 {
		score = 1.0
	}

	return score
}

// computeDensityScore measures connection density.
// Too sparse = underutilized capacity
// Too dense = may overfit
// We prefer moderate density close to target.
func (p *ProxyEvaluator) computeDensityScore(arch *searchspace.Architecture) float64 {
	// Count non-zero (active) edges
	activeEdges := 0
	totalEdges := 0

	for _, cell := range []*searchspace.Cell{arch.NormalCell, arch.ReductionCell} {
		for _, node := range cell.Nodes {
			for _, edge := range node.Edges {
				totalEdges++
				if edge.Operation != searchspace.OpZero && edge.Operation != searchspace.OpNone {
					activeEdges++
				}
			}
		}
	}

	if totalEdges == 0 {
		return 0
	}

	density := float64(activeEdges) / float64(totalEdges)

	// Score based on distance from target (closer = better)
	diff := math.Abs(density - p.config.TargetDensity)
	score := 1.0 - diff

	if score < 0 {
		score = 0
	}

	return score
}

// computeSkipRatioScore measures the ratio of skip connections.
// Skip connections (identity) help gradient flow but too many
// limit the model's representational power.
func (p *ProxyEvaluator) computeSkipRatioScore(arch *searchspace.Architecture) float64 {
	skipCount := 0
	totalOps := 0

	for _, cell := range []*searchspace.Cell{arch.NormalCell, arch.ReductionCell} {
		for _, node := range cell.Nodes {
			for _, edge := range node.Edges {
				if edge.Operation != searchspace.OpNone && edge.Operation != searchspace.OpZero {
					totalOps++
					if edge.Operation == searchspace.OpIdentity {
						skipCount++
					}
				}
			}
		}
	}

	if totalOps == 0 {
		return 0
	}

	skipRatio := float64(skipCount) / float64(totalOps)

	// Score based on distance from target
	diff := math.Abs(skipRatio - p.config.TargetSkipRatio)
	score := 1.0 - (diff * 2) // Penalize deviation more strongly

	if score < 0 {
		score = 0
	}

	return score
}
