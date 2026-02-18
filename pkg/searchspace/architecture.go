package searchspace

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/google/uuid"
)

// Architecture represents a complete neural architecture.
// It consists of a normal cell (repeated) and a reduction cell.
// The full network is built by stacking these cells.
//
// Network structure (for ImageNet-style):
// [Stem] -> [Normal×N] -> [Reduction] -> [Normal×N] -> [Reduction] -> [Normal×N] -> [Head]
//
// The stem handles initial dimensionality, cells do the heavy lifting,
// and the head produces final predictions.
type Architecture struct {
	// ID is a unique identifier for this architecture.
	// Using UUID to ensure uniqueness across distributed searches.
	ID string `json:"id"`

	// NormalCell defines the structure of normal cells.
	// Normal cells preserve spatial dimensions (stride=1).
	NormalCell *Cell `json:"normal_cell"`

	// ReductionCell defines the structure of reduction cells.
	// Reduction cells halve spatial dimensions (stride=2).
	ReductionCell *Cell `json:"reduction_cell"`

	// Metadata stores additional information about this architecture.
	Metadata ArchitectureMetadata `json:"metadata"`
}

// ArchitectureMetadata stores non-structural information about an architecture.
// This is separated from the core structure to keep the architecture
// definition clean and allow metadata to be optional.
type ArchitectureMetadata struct {
	// CreatedAt is when this architecture was generated.
	CreatedAt time.Time `json:"created_at"`

	// Generation is the evolutionary generation (for evolutionary search).
	// -1 or 0 for non-evolutionary methods.
	Generation int `json:"generation"`

	// ParentID is the ID of the parent architecture (for mutation-based search).
	// Empty string for initial random architectures.
	ParentID string `json:"parent_id,omitempty"`

	// MutationType describes what mutation created this architecture.
	// Empty for random/initial architectures.
	MutationType string `json:"mutation_type,omitempty"`

	// Fitness stores the evaluated fitness/accuracy of this architecture.
	// NaN or 0 if not yet evaluated.
	Fitness float64 `json:"fitness"`

	// EvaluationTime is how long evaluation took (for analysis).
	EvaluationTime time.Duration `json:"evaluation_time,omitempty"`

	// Notes can store any additional information.
	Notes string `json:"notes,omitempty"`
}

// NewArchitecture creates a new architecture with the given cells.
// This is the primary constructor for creating architectures.
//
// Parameters:
//   - normalCell: the cell structure for normal (non-reducing) cells
//   - reductionCell: the cell structure for reduction cells
//
// Returns:
//   - A pointer to a new Architecture with a unique ID
//
// Example:
//
//	normal := NewCell(NormalCell, 4, 2, 2)
//	reduction := NewCell(ReductionCell, 4, 2, 2)
//	arch := NewArchitecture(normal, reduction)
func NewArchitecture(normalCell, reductionCell *Cell) *Architecture {
	return &Architecture{
		ID:            uuid.New().String(),
		NormalCell:    normalCell,
		ReductionCell: reductionCell,
		Metadata: ArchitectureMetadata{
			CreatedAt:  time.Now(),
			Generation: 0,
		},
	}
}

// Clone creates a deep copy of the architecture.
// Essential for evolutionary algorithms where offspring are mutated
// copies of parents.
func (a *Architecture) Clone() *Architecture {
	clone := &Architecture{
		ID:            uuid.New().String(), // New ID for clone
		NormalCell:    a.NormalCell.Clone(),
		ReductionCell: a.ReductionCell.Clone(),
		Metadata:      a.Metadata, // Copy metadata (don't need deep copy)
	}
	clone.Metadata.ParentID = a.ID
	clone.Metadata.CreatedAt = time.Now()
	return clone
}

// Hash returns a deterministic hash of the architecture structure.
// Used for deduplication: if two architectures have the same hash,
// they are structurally identical.
//
// Note: This only hashes the cell structures, not metadata.
// Two architectures with different fitness but same structure
// will have the same hash.
func (a *Architecture) Hash() string {
	// Combine hashes of both cells
	return a.NormalCell.Hash() + "_" + a.ReductionCell.Hash()
}

// IsValid checks if the architecture is valid.
// An architecture is valid if both cells are valid.
func (a *Architecture) IsValid() bool {
	return a.NormalCell != nil &&
		a.ReductionCell != nil &&
		a.NormalCell.IsValid() &&
		a.ReductionCell.IsValid()
}

// ParameterEstimate returns a rough estimate of total parameters.
// The actual count depends on:
// - Number of cell repeats (numCells)
// - Base channel count (channels)
// - Stem and head architecture
//
// This estimate only considers the relative complexity of the cells.
// Multiply by channels² for a better estimate.
func (a *Architecture) ParameterEstimate() int {
	// Assume typical configuration: 6 normal cells per stage, 2 reduction cells
	normalParams := a.NormalCell.ParameterEstimate() * 6 * 3 // 3 stages
	reductionParams := a.ReductionCell.ParameterEstimate() * 2
	return normalParams + reductionParams
}

// ToJSON serializes the architecture to JSON bytes.
// Parameters:
//   - pretty: if true, use indented formatting
//
// Returns:
//   - JSON bytes
//   - Error if serialization fails
func (a *Architecture) ToJSON(pretty bool) ([]byte, error) {
	if pretty {
		return json.MarshalIndent(a, "", "  ")
	}
	return json.Marshal(a)
}

// SaveJSON saves the architecture to a JSON file.
// Parameters:
//   - path: file path to save to
//
// Returns:
//   - Error if file operations fail
func (a *Architecture) SaveJSON(path string) error {
	data, err := a.ToJSON(true)
	if err != nil {
		return fmt.Errorf("marshaling architecture: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("writing file %s: %w", path, err)
	}
	return nil
}

// ArchitectureFromJSON deserializes an architecture from JSON bytes.
// Parameters:
//   - data: JSON bytes
//
// Returns:
//   - The parsed architecture
//   - Error if parsing fails or the result is invalid
func ArchitectureFromJSON(data []byte) (*Architecture, error) {
	var arch Architecture
	if err := json.Unmarshal(data, &arch); err != nil {
		return nil, fmt.Errorf("unmarshaling architecture: %w", err)
	}
	if !arch.IsValid() {
		return nil, fmt.Errorf("invalid architecture structure")
	}
	return &arch, nil
}

// LoadArchitectureJSON loads an architecture from a JSON file.
// Parameters:
//   - path: file path to load from
//
// Returns:
//   - The loaded architecture
//   - Error if file operations or parsing fails
func LoadArchitectureJSON(path string) (*Architecture, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading file %s: %w", path, err)
	}
	return ArchitectureFromJSON(data)
}

// Genotype is an alternative encoding of an architecture as a flat slice.
// This is useful for evolutionary algorithms that operate on flat vectors.
// This encoding is compatible with the NAS-Bench-201 format.
//
// Format: for each node (in order), for each edge (in order):
//
//	[input_node_idx, operation_idx, input_node_idx, operation_idx, ...]
//
// The slice length is: num_intermediate_nodes × edges_per_node × 2
type Genotype []int

// ToGenotype converts the architecture to a flat genotype representation.
// This is useful for:
// - Crossover operations (swap slices between parents)
// - Mutation operations (flip individual values)
// - Storage in fixed-size arrays
//
// Returns:
//   - normalGenotype: flat representation of normal cell
//   - reductionGenotype: flat representation of reduction cell
func (a *Architecture) ToGenotype() (normalGenotype, reductionGenotype Genotype) {
	normalGenotype = cellToGenotype(a.NormalCell)
	reductionGenotype = cellToGenotype(a.ReductionCell)
	return
}

// cellToGenotype converts a single cell to genotype format.
func cellToGenotype(c *Cell) Genotype {
	// Calculate size: 2 values per edge (input + op)
	totalEdges := 0
	for _, node := range c.Nodes {
		totalEdges += len(node.Edges)
	}

	genotype := make(Genotype, totalEdges*2)
	idx := 0
	for _, node := range c.Nodes {
		for _, edge := range node.Edges {
			genotype[idx] = edge.InputNode
			genotype[idx+1] = int(edge.Operation)
			idx += 2
		}
	}
	return genotype
}

// FromGenotype reconstructs cells from genotype representations.
// This is the inverse of ToGenotype.
//
// Parameters:
//   - normalGenotype: flat representation of normal cell
//   - reductionGenotype: flat representation of reduction cell
//   - numNodes: number of intermediate nodes per cell
//   - numInputNodes: number of input nodes per cell
//   - edgesPerNode: number of edges per node
//
// Returns:
//   - A new architecture with cells reconstructed from genotypes
//   - Error if genotype length doesn't match expected size
func FromGenotype(normalGenotype, reductionGenotype Genotype,
	numNodes, numInputNodes, edgesPerNode int) (*Architecture, error) {

	expectedLen := numNodes * edgesPerNode * 2
	if len(normalGenotype) != expectedLen || len(reductionGenotype) != expectedLen {
		return nil, fmt.Errorf("genotype length mismatch: expected %d, got normal=%d, reduction=%d",
			expectedLen, len(normalGenotype), len(reductionGenotype))
	}

	normalCell, err := genotypeToCell(normalGenotype, NormalCell, numNodes, numInputNodes, edgesPerNode)
	if err != nil {
		return nil, fmt.Errorf("parsing normal cell genotype: %w", err)
	}

	reductionCell, err := genotypeToCell(reductionGenotype, ReductionCell, numNodes, numInputNodes, edgesPerNode)
	if err != nil {
		return nil, fmt.Errorf("parsing reduction cell genotype: %w", err)
	}

	return NewArchitecture(normalCell, reductionCell), nil
}

// genotypeToCell converts genotype back to a Cell structure.
func genotypeToCell(genotype Genotype, cellType CellType,
	numNodes, numInputNodes, edgesPerNode int) (*Cell, error) {

	cell := NewCell(cellType, numNodes, numInputNodes, edgesPerNode)
	idx := 0
	for i := range cell.Nodes {
		for j := range cell.Nodes[i].Edges {
			if idx+1 >= len(genotype) {
				return nil, fmt.Errorf("genotype too short at index %d", idx)
			}
			cell.Nodes[i].Edges[j].InputNode = genotype[idx]
			cell.Nodes[i].Edges[j].Operation = OperationType(genotype[idx+1])
			idx += 2
		}
	}
	return cell, nil
}

// Summary returns a human-readable summary of the architecture.
func (a *Architecture) Summary() string {
	return fmt.Sprintf(
		"Architecture %s:\n"+
			"  Normal Cell:    %d nodes, %d params (est)\n"+
			"  Reduction Cell: %d nodes, %d params (est)\n"+
			"  Fitness:        %.4f\n"+
			"  Created:        %s",
		a.ID[:8],
		len(a.NormalCell.Nodes),
		a.NormalCell.ParameterEstimate(),
		len(a.ReductionCell.Nodes),
		a.ReductionCell.ParameterEstimate(),
		a.Metadata.Fitness,
		a.Metadata.CreatedAt.Format("2006-01-02 15:04:05"),
	)
}
