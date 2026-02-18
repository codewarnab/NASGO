package searchspace

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

// Edge represents a directed connection between two nodes in a cell.
// Each edge carries both a source node and an operation to apply.
//
// In the NAS context, an edge represents:
// 1. WHERE to get input from (InputNode)
// 2. WHAT to do with that input (Operation)
//
// Example: Edge{InputNode: 0, Operation: OpConv3x3}
// means "take output from node 0, apply 3x3 conv"
type Edge struct {
	// InputNode is the index of the source node.
	// Nodes 0 and 1 are typically the cell inputs (previous cell outputs).
	// Nodes 2+ are intermediate nodes within the cell.
	InputNode int `json:"input_node"`

	// Operation is the operation to apply to the input.
	Operation OperationType `json:"operation"`
}

// String returns a human-readable representation of the edge.
func (e Edge) String() string {
	return fmt.Sprintf("%d->%s", e.InputNode, e.Operation)
}

// Node represents an intermediate node within a cell.
// Each node aggregates inputs from multiple edges.
//
// In most NAS papers, each node receives exactly 2 inputs,
// which are then summed (element-wise addition).
// This allows for skip connections and multi-path topologies.
//
// The output of a node is: node_output = op1(input1) + op2(input2)
type Node struct {
	// Edges are the incoming connections to this node.
	// Typically exactly 2 edges per node (following NASNet convention).
	// Each edge specifies which earlier node to take input from
	// and what operation to apply.
	Edges []Edge `json:"edges"`
}

// String returns a human-readable representation of the node.
func (n Node) String() string {
	parts := make([]string, len(n.Edges))
	for i, e := range n.Edges {
		parts[i] = e.String()
	}
	return fmt.Sprintf("node(%s)", join(parts, " + "))
}

// join is a helper to join strings (avoiding strings import for brevity).
func join(parts []string, sep string) string {
	if len(parts) == 0 {
		return ""
	}
	result := parts[0]
	for _, p := range parts[1:] {
		result += sep + p
	}
	return result
}

// CellType distinguishes between normal and reduction cells.
// This is a key concept from NASNet paper.
type CellType int

const (
	// NormalCell preserves spatial dimensions (height, width).
	// Stride = 1 for all operations.
	NormalCell CellType = iota

	// ReductionCell reduces spatial dimensions by half.
	// Stride = 2 for operations connected to input nodes.
	// This is where "pooling" happens in the network.
	ReductionCell
)

// String returns the human-readable name of the cell type.
func (c CellType) String() string {
	if c == NormalCell {
		return "normal"
	}
	return "reduction"
}

// Cell represents a single cell (micro-architecture) in the network.
// A cell is a directed acyclic graph (DAG) where:
// - Input nodes (0, 1) receive outputs from previous cells
// - Intermediate nodes (2, 3, ...) apply operations and aggregate
// - Output is the concatenation of all intermediate node outputs
//
// The full network is built by stacking multiple cells:
// [Input] -> [Cell] -> [Cell] -> [Reduction Cell] -> [Cell] -> ... -> [Output]
//
// This structure makes the search space tractable because:
// 1. We only search for a small cell, not the full network
// 2. Cells are reusable (stack the same cell multiple times)
// 3. Transfer learning: cells found on CIFAR-10 often work on ImageNet
type Cell struct {
	// Type indicates if this is a normal or reduction cell.
	Type CellType `json:"type"`

	// Nodes are the intermediate nodes in the cell (not including input nodes).
	// Typically 4-7 nodes. More nodes = more complex cell = larger search space.
	// The NASNet paper uses 5 intermediate nodes.
	Nodes []Node `json:"nodes"`

	// NumInputNodes is how many input nodes this cell has (typically 2).
	// Input nodes are indexed 0 to NumInputNodes-1.
	// They receive outputs from the previous two cells.
	NumInputNodes int `json:"num_input_nodes"`
}

// NewCell creates a new cell with the specified configuration.
// This is the constructor function (Go convention: New<TypeName>).
//
// Parameters:
//   - cellType: NormalCell or ReductionCell
//   - numNodes: number of intermediate nodes (not including input nodes)
//   - numInputNodes: number of input nodes (typically 2)
//   - edgesPerNode: number of edges per node (typically 2)
//
// Returns:
//   - A pointer to a new Cell with all nodes initialized with zero operations
//
// Example:
//
//	cell := NewCell(NormalCell, 4, 2, 2)
//	// Creates a normal cell with 4 intermediate nodes, 2 inputs, 2 edges each
func NewCell(cellType CellType, numNodes, numInputNodes, edgesPerNode int) *Cell {
	nodes := make([]Node, numNodes)
	for i := range nodes {
		nodes[i].Edges = make([]Edge, edgesPerNode)
		// Edges are zero-initialized, which means:
		// - InputNode: 0
		// - Operation: OpNone (0)
	}
	return &Cell{
		Type:          cellType,
		Nodes:         nodes,
		NumInputNodes: numInputNodes,
	}
}

// Clone creates a deep copy of the cell.
// Important for evolutionary algorithms where we mutate offspring
// without affecting the parent.
func (c *Cell) Clone() *Cell {
	clone := &Cell{
		Type:          c.Type,
		NumInputNodes: c.NumInputNodes,
		Nodes:         make([]Node, len(c.Nodes)),
	}
	for i, node := range c.Nodes {
		clone.Nodes[i].Edges = make([]Edge, len(node.Edges))
		copy(clone.Nodes[i].Edges, node.Edges)
	}
	return clone
}

// TotalNodes returns the total number of nodes (input + intermediate).
func (c *Cell) TotalNodes() int {
	return c.NumInputNodes + len(c.Nodes)
}

// NodeIndex returns the global index for an intermediate node.
// Intermediate nodes are numbered after input nodes.
//
// Parameters:
//   - intermediateIdx: the index within c.Nodes (0 to len(Nodes)-1)
//
// Returns:
//   - The global node index (used for edge connections)
//
// Example:
//
//	cell := NewCell(NormalCell, 4, 2, 2)
//	idx := cell.NodeIndex(0) // Returns 2 (first intermediate node)
//	idx := cell.NodeIndex(1) // Returns 3 (second intermediate node)
func (c *Cell) NodeIndex(intermediateIdx int) int {
	return c.NumInputNodes + intermediateIdx
}

// ValidInputsForNode returns all valid input node indices for a given node.
// A node can only connect to EARLIER nodes (DAG constraint).
// This prevents cycles and ensures the cell is well-defined.
//
// Parameters:
//   - nodeIdx: the global index of the node (including input nodes)
//
// Returns:
//   - A slice of valid input indices: [0, 1, ..., nodeIdx-1]
func (c *Cell) ValidInputsForNode(nodeIdx int) []int {
	if nodeIdx <= 0 {
		return nil
	}
	inputs := make([]int, nodeIdx)
	for i := 0; i < nodeIdx; i++ {
		inputs[i] = i
	}
	return inputs
}

// Hash returns a deterministic hash of the cell structure.
// Used for deduplication in the search algorithm.
// Two cells with identical structure will have the same hash.
func (c *Cell) Hash() string {
	// Use JSON for deterministic serialization
	// (Note: in production, consider a more efficient binary format)
	data, _ := json.Marshal(c)
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:8]) // First 8 bytes is enough
}

// IsValid checks if the cell structure is valid.
// A cell is valid if:
// 1. All edge input nodes are within valid range
// 2. All edge operations are valid
// 3. No node connects to itself or future nodes
//
// Returns:
//   - true if the cell is valid
//   - false if any validation check fails
func (c *Cell) IsValid() bool {
	for i, node := range c.Nodes {
		globalIdx := c.NodeIndex(i)
		for _, edge := range node.Edges {
			// Check input is a valid earlier node
			if edge.InputNode < 0 || edge.InputNode >= globalIdx {
				return false
			}
			// Check operation is valid (OpZero is valid - means no connection)
			if edge.Operation < OpNone || edge.Operation >= opCount {
				return false
			}
		}
	}
	return true
}

// ParameterEstimate returns a rough estimate of the cell's parameter count.
// Useful for constraining the search to find efficient architectures.
// The actual count depends on the channel configuration at runtime.
func (c *Cell) ParameterEstimate() int {
	total := 0
	for _, node := range c.Nodes {
		for _, edge := range node.Edges {
			total += edge.Operation.ParameterCount()
		}
	}
	return total
}

// TopologicalOrder returns the order in which nodes should be computed.
// For our DAG structure with sequential node indices, this is simply [0, 1, 2, ...].
// More complex DAGs would require actual topological sorting.
func (c *Cell) TopologicalOrder() []int {
	order := make([]int, c.TotalNodes())
	for i := range order {
		order[i] = i
	}
	return order
}

// UsedOperations returns the set of operations used in this cell.
// Useful for analysis and reporting.
func (c *Cell) UsedOperations() []OperationType {
	seen := make(map[OperationType]bool)
	for _, node := range c.Nodes {
		for _, edge := range node.Edges {
			if edge.Operation.IsValid() {
				seen[edge.Operation] = true
			}
		}
	}

	ops := make([]OperationType, 0, len(seen))
	for op := range seen {
		ops = append(ops, op)
	}
	// Sort for deterministic output
	sort.Slice(ops, func(i, j int) bool {
		return ops[i] < ops[j]
	})
	return ops
}
