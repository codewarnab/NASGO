package searchspace

import (
	"fmt"
	"math/rand"
)

// SearchSpace defines the configuration for the neural architecture search space.
// It specifies:
// - What operations are available at each edge
// - How many nodes cells have
// - How the search should generate and mutate architectures
//
// The search space design is crucial for NAS success:
// - Too small: may miss good architectures
// - Too large: search becomes intractable
// - Good design: captures the space of "reasonable" architectures
type SearchSpace struct {
	// Operations are the available operation types at each edge.
	// Typically includes convolutions, pooling, skip connections.
	// Including OpZero allows the search to "remove" edges.
	Operations []OperationType `json:"operations"`

	// NumNodes is the number of intermediate nodes per cell.
	// More nodes = more complex cells = larger search space.
	// Typical values: 4 (DARTS), 5 (NASNet), 7 (AmoebaNet).
	NumNodes int `json:"num_nodes"`

	// NumInputNodes is the number of input nodes per cell.
	// Typically 2: outputs from the previous two cells.
	// This enables residual connections across cells.
	NumInputNodes int `json:"num_input_nodes"`

	// EdgesPerNode is how many incoming edges each node has.
	// Typically 2: each node aggregates two transformed inputs.
	// More edges = more complex nodes = larger search space.
	EdgesPerNode int `json:"edges_per_node"`

	// rng is the random number generator for sampling.
	// Using a seeded RNG enables reproducibility.
	rng *rand.Rand
}

// DefaultSearchSpace returns the default search space configuration.
// This matches the DARTS paper configuration.
//
// Reference: DARTS paper, Section 3.1
func DefaultSearchSpace() *SearchSpace {
	return &SearchSpace{
		Operations:    DefaultOperations(),
		NumNodes:      4,
		NumInputNodes: 2,
		EdgesPerNode:  2,
		rng:           rand.New(rand.NewSource(42)),
	}
}

// NewSearchSpace creates a search space with custom configuration.
// Parameters:
//   - operations: available operations at each edge
//   - numNodes: intermediate nodes per cell
//   - numInputNodes: input nodes per cell
//   - edgesPerNode: edges per node
//   - seed: random seed for reproducibility (-1 for random seed)
//
// Returns:
//   - Configured SearchSpace
//   - Error if configuration is invalid
func NewSearchSpace(operations []OperationType, numNodes, numInputNodes, edgesPerNode int, seed int64) (*SearchSpace, error) {
	// Validate configuration
	if len(operations) == 0 {
		return nil, fmt.Errorf("operations cannot be empty")
	}
	if numNodes < 1 {
		return nil, fmt.Errorf("numNodes must be at least 1, got %d", numNodes)
	}
	if numInputNodes < 1 {
		return nil, fmt.Errorf("numInputNodes must be at least 1, got %d", numInputNodes)
	}
	if edgesPerNode < 1 {
		return nil, fmt.Errorf("edgesPerNode must be at least 1, got %d", edgesPerNode)
	}

	// Use random seed if -1
	if seed == -1 {
		seed = rand.Int63()
	}

	return &SearchSpace{
		Operations:    operations,
		NumNodes:      numNodes,
		NumInputNodes: numInputNodes,
		EdgesPerNode:  edgesPerNode,
		rng:           rand.New(rand.NewSource(seed)),
	}, nil
}

// SetSeed resets the random number generator with a new seed.
// Call this before sampling to ensure reproducibility.
func (s *SearchSpace) SetSeed(seed int64) {
	s.rng = rand.New(rand.NewSource(seed))
}

// Size returns the total number of possible architectures in this search space.
// This can be astronomically large!
//
// Formula for one cell:
// For each intermediate node (i = numInputNodes to numInputNodes + numNodes - 1):
//   - Choose 2 input nodes from i available nodes: C(i, 2) = i*(i-1)/2 ways
//   - But we pick with replacement and order doesn't matter for edges
//   - Actually: we pick edgesPerNode inputs, each from [0, i) nodes
//   - Each edge has len(operations) possible operations
//
// Simplified calculation (assuming edges can pick same node):
// For node i (0-indexed intermediate), there are (i + numInputNodes) input choices
// Product over all nodes and edges, times operation choices
//
// Returns the size as a float64 since it can exceed int64 max.
func (s *SearchSpace) Size() float64 {
	numOps := float64(len(s.Operations))
	var cellSize float64 = 1

	for i := 0; i < s.NumNodes; i++ {
		nodeIdx := s.NumInputNodes + i // Global index of this intermediate node
		numInputChoices := float64(nodeIdx)

		// For each edge: choose input node AND operation
		for j := 0; j < s.EdgesPerNode; j++ {
			cellSize *= numInputChoices * numOps
		}
	}

	// Total is normal_cell_size * reduction_cell_size
	// (assuming they share the same structure)
	return cellSize * cellSize
}

// SampleRandomArchitecture generates a completely random architecture.
// Every edge randomly picks:
//
//  1. An input from valid earlier nodes
//  2. An operation from available operations
//
// This is the baseline for random search and initial population generation.
func (s *SearchSpace) SampleRandomArchitecture() *Architecture {
	normalCell := s.sampleRandomCell(NormalCell)
	reductionCell := s.sampleRandomCell(ReductionCell)
	return NewArchitecture(normalCell, reductionCell)
}

// sampleRandomCell creates a random cell of the specified type.
func (s *SearchSpace) sampleRandomCell(cellType CellType) *Cell {
	cell := NewCell(cellType, s.NumNodes, s.NumInputNodes, s.EdgesPerNode)

	for i := range cell.Nodes {
		nodeIdx := cell.NodeIndex(i)
		validInputs := cell.ValidInputsForNode(nodeIdx)

		for j := range cell.Nodes[i].Edges {
			// Random input node from valid earlier nodes
			inputIdx := s.rng.Intn(len(validInputs))
			cell.Nodes[i].Edges[j].InputNode = validInputs[inputIdx]

			// Random operation
			opIdx := s.rng.Intn(len(s.Operations))
			cell.Nodes[i].Edges[j].Operation = s.Operations[opIdx]
		}
	}

	return cell
}

// MutationType defines the types of mutations we can apply.
type MutationType int

const (
	// MutateOperation changes one edge's operation.
	// Keeps the same input node, changes what operation is applied.
	MutateOperation MutationType = iota

	// MutateInput changes one edge's input node.
	// Keeps the same operation, changes where input comes from.
	MutateInput

	// MutateEdge changes both operation and input of one edge.
	MutateEdge
)

// Mutate creates a mutated copy of the architecture.
// The mutation type is randomly selected.
//
// Parameters:
//   - arch: the parent architecture to mutate
//
// Returns:
//   - A new architecture that is a mutated copy of the parent
//
// Mutation strategy:
// 1. Clone the parent
// 2. Randomly select which cell to mutate (normal or reduction)
// 3. Randomly select which node and edge to mutate
// 4. Apply random mutation (change operation and/or input)
func (s *SearchSpace) Mutate(arch *Architecture) *Architecture {
	child := arch.Clone()
	child.Metadata.ParentID = arch.ID

	// Randomly pick which cell to mutate
	var cell *Cell
	if s.rng.Float32() < 0.5 {
		cell = child.NormalCell
		child.Metadata.MutationType = "normal_cell"
	} else {
		cell = child.ReductionCell
		child.Metadata.MutationType = "reduction_cell"
	}

	// Randomly pick which node to mutate
	nodeIdx := s.rng.Intn(len(cell.Nodes))
	node := &cell.Nodes[nodeIdx]
	globalNodeIdx := cell.NodeIndex(nodeIdx)

	// Randomly pick which edge to mutate
	edgeIdx := s.rng.Intn(len(node.Edges))
	edge := &node.Edges[edgeIdx]

	// Randomly pick mutation type
	mutationType := MutationType(s.rng.Intn(3))

	switch mutationType {
	case MutateOperation:
		// Change operation only
		newOpIdx := s.rng.Intn(len(s.Operations))
		edge.Operation = s.Operations[newOpIdx]
		child.Metadata.MutationType += "_op"

	case MutateInput:
		// Change input only
		validInputs := cell.ValidInputsForNode(globalNodeIdx)
		newInputIdx := s.rng.Intn(len(validInputs))
		edge.InputNode = validInputs[newInputIdx]
		child.Metadata.MutationType += "_input"

	case MutateEdge:
		// Change both
		validInputs := cell.ValidInputsForNode(globalNodeIdx)
		newInputIdx := s.rng.Intn(len(validInputs))
		edge.InputNode = validInputs[newInputIdx]

		newOpIdx := s.rng.Intn(len(s.Operations))
		edge.Operation = s.Operations[newOpIdx]
		child.Metadata.MutationType += "_edge"
	}

	return child
}

// Crossover creates a child architecture from two parents.
// Uses single-point crossover on the genotype representation.
//
// Parameters:
//   - parent1, parent2: the parent architectures
//
// Returns:
//   - A child architecture combining parts of both parents
//
// Strategy:
// - For normal cell: pick crossover point, take first half from parent1, second from parent2
// - For reduction cell: same approach
func (s *SearchSpace) Crossover(parent1, parent2 *Architecture) *Architecture {
	normal1, reduction1 := parent1.ToGenotype()
	normal2, reduction2 := parent2.ToGenotype()

	// Single-point crossover for normal cell
	childNormal := s.crossoverGenotype(normal1, normal2)
	childReduction := s.crossoverGenotype(reduction1, reduction2)

	child, _ := FromGenotype(childNormal, childReduction,
		s.NumNodes, s.NumInputNodes, s.EdgesPerNode)
	child.Metadata.ParentID = parent1.ID
	child.Metadata.MutationType = "crossover"
	child.Metadata.Notes = fmt.Sprintf("parents: %s, %s", parent1.ID[:8], parent2.ID[:8])

	return child
}

// crossoverGenotype performs single-point crossover on two genotypes.
func (s *SearchSpace) crossoverGenotype(g1, g2 Genotype) Genotype {
	if len(g1) == 0 || len(g2) == 0 {
		return g1
	}

	// Pick crossover point (must be at edge boundary: every 2 values)
	numEdges := len(g1) / 2
	crossPoint := s.rng.Intn(numEdges) * 2

	child := make(Genotype, len(g1))
	copy(child[:crossPoint], g1[:crossPoint])
	copy(child[crossPoint:], g2[crossPoint:])
	return child
}

// SampleNeighbor generates a random neighbor of the given architecture.
// A neighbor differs by exactly one mutation.
// This is useful for local search algorithms like hill climbing.
func (s *SearchSpace) SampleNeighbor(arch *Architecture) *Architecture {
	return s.Mutate(arch)
}

// PopulateInitial generates an initial population of random architectures.
// Used to initialize evolutionary algorithms.
//
// Parameters:
//   - size: number of architectures to generate
//
// Returns:
//   - Slice of random architectures
func (s *SearchSpace) PopulateInitial(size int) []*Architecture {
	population := make([]*Architecture, size)
	for i := range population {
		population[i] = s.SampleRandomArchitecture()
		population[i].Metadata.Generation = 0
	}
	return population
}

// Validate checks if an architecture is valid within this search space.
// An architecture is valid if:
// - All operations are in the allowed set
// - All input connections are within valid ranges
//
// Parameters:
//   - arch: the architecture to validate
//
// Returns:
//   - nil if valid
//   - Error describing the validation failure otherwise
func (s *SearchSpace) Validate(arch *Architecture) error {
	opSet := make(map[OperationType]bool)
	for _, op := range s.Operations {
		opSet[op] = true
	}

	// Validate both cells
	cells := []*Cell{arch.NormalCell, arch.ReductionCell}
	cellNames := []string{"normal", "reduction"}

	for c, cell := range cells {
		for i, node := range cell.Nodes {
			globalIdx := cell.NodeIndex(i)
			for j, edge := range node.Edges {
				// Check operation is allowed
				if !opSet[edge.Operation] && edge.Operation != OpNone {
					return fmt.Errorf("%s cell node %d edge %d: operation %s not in search space",
						cellNames[c], i, j, edge.Operation)
				}
				// Check input is valid
				if edge.InputNode < 0 || edge.InputNode >= globalIdx {
					return fmt.Errorf("%s cell node %d edge %d: invalid input node %d (must be < %d)",
						cellNames[c], i, j, edge.InputNode, globalIdx)
				}
			}
		}
	}
	return nil
}

// String returns a human-readable description of the search space.
func (s *SearchSpace) String() string {
	opNames := make([]string, len(s.Operations))
	for i, op := range s.Operations {
		opNames[i] = op.String()
	}
	return fmt.Sprintf(
		"SearchSpace(nodes=%d, inputs=%d, edges=%d, ops=%v, size=%.2e)",
		s.NumNodes, s.NumInputNodes, s.EdgesPerNode, opNames, s.Size(),
	)
}
