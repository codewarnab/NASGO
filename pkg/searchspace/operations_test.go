package searchspace

import (
	"testing"
)

// ─── OperationType Tests ────────────────────────────────────────────────────

func TestOperationTypeString(t *testing.T) {
	tests := []struct {
		op   OperationType
		want string
	}{
		{OpNone, "none"},
		{OpIdentity, "identity"},
		{OpConv1x1, "conv_1x1"},
		{OpConv3x3, "conv_3x3"},
		{OpConv5x5, "conv_5x5"},
		{OpConv7x7, "conv_7x7"},
		{OpSepConv3x3, "sep_conv_3x3"},
		{OpSepConv5x5, "sep_conv_5x5"},
		{OpDilConv3x3, "dil_conv_3x3"},
		{OpDilConv5x5, "dil_conv_5x5"},
		{OpMaxPool3x3, "max_pool_3x3"},
		{OpAvgPool3x3, "avg_pool_3x3"},
		{OpZero, "zero"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got := tt.op.String()
			if got != tt.want {
				t.Errorf("OperationType(%d).String() = %q, want %q", tt.op, got, tt.want)
			}
		})
	}
}

func TestOperationTypeStringUnknown(t *testing.T) {
	op := OperationType(-1)
	got := op.String()
	if got == "" {
		t.Error("expected non-empty string for unknown operation")
	}

	op2 := OperationType(999)
	got2 := op2.String()
	if got2 == "" {
		t.Error("expected non-empty string for out-of-range operation")
	}
}

func TestOperationTypeIsValid(t *testing.T) {
	tests := []struct {
		op   OperationType
		want bool
	}{
		{OpNone, false},
		{OpIdentity, true},
		{OpConv3x3, true},
		{OpZero, true},
		{OperationType(-1), false},
		{OperationType(999), false},
	}

	for _, tt := range tests {
		t.Run(tt.op.String(), func(t *testing.T) {
			if got := tt.op.IsValid(); got != tt.want {
				t.Errorf("OperationType(%d).IsValid() = %v, want %v", tt.op, got, tt.want)
			}
		})
	}
}

func TestOperationTypeIsPooling(t *testing.T) {
	pooling := []OperationType{OpMaxPool3x3, OpAvgPool3x3}
	notPooling := []OperationType{OpNone, OpIdentity, OpConv3x3, OpSepConv3x3, OpZero}

	for _, op := range pooling {
		if !op.IsPooling() {
			t.Errorf("expected %s to be pooling", op)
		}
	}
	for _, op := range notPooling {
		if op.IsPooling() {
			t.Errorf("expected %s to NOT be pooling", op)
		}
	}
}

func TestOperationTypeIsConvolution(t *testing.T) {
	convs := []OperationType{
		OpConv1x1, OpConv3x3, OpConv5x5, OpConv7x7,
		OpSepConv3x3, OpSepConv5x5, OpDilConv3x3, OpDilConv5x5,
	}
	notConvs := []OperationType{OpNone, OpIdentity, OpMaxPool3x3, OpAvgPool3x3, OpZero}

	for _, op := range convs {
		if !op.IsConvolution() {
			t.Errorf("expected %s to be convolution", op)
		}
	}
	for _, op := range notConvs {
		if op.IsConvolution() {
			t.Errorf("expected %s to NOT be convolution", op)
		}
	}
}

func TestOperationTypeParameterCount(t *testing.T) {
	// Operations with zero parameters
	zeroParams := []OperationType{OpNone, OpZero, OpIdentity, OpMaxPool3x3, OpAvgPool3x3}
	for _, op := range zeroParams {
		if got := op.ParameterCount(); got != 0 {
			t.Errorf("%s.ParameterCount() = %d, want 0", op, got)
		}
	}

	// Conv3x3 should have 9 params
	if got := OpConv3x3.ParameterCount(); got != 9 {
		t.Errorf("OpConv3x3.ParameterCount() = %d, want 9", got)
	}

	// Conv5x5 should have 25 params
	if got := OpConv5x5.ParameterCount(); got != 25 {
		t.Errorf("OpConv5x5.ParameterCount() = %d, want 25", got)
	}
}

func TestParseOperationType(t *testing.T) {
	tests := []struct {
		input string
		want  OperationType
		err   bool
	}{
		{"conv_3x3", OpConv3x3, false},
		{"CONV_3X3", OpConv3x3, false},
		{" conv_3x3 ", OpConv3x3, false},
		{"identity", OpIdentity, false},
		{"zero", OpZero, false},
		{"sep_conv_3x3", OpSepConv3x3, false},
		{"max_pool_3x3", OpMaxPool3x3, false},
		{"invalid_op", OpNone, true},
		{"", OpNone, true},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := ParseOperationType(tt.input)
			if (err != nil) != tt.err {
				t.Errorf("ParseOperationType(%q) error = %v, wantErr %v", tt.input, err, tt.err)
				return
			}
			if got != tt.want {
				t.Errorf("ParseOperationType(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestAllOperations(t *testing.T) {
	ops := AllOperations()

	// Should include all valid operations (OpIdentity through OpZero)
	if len(ops) == 0 {
		t.Fatal("AllOperations() returned empty slice")
	}

	// Should not include OpNone
	for _, op := range ops {
		if op == OpNone {
			t.Error("AllOperations() should not include OpNone")
		}
	}

	// Should include common operations
	hasConv3x3 := false
	hasIdentity := false
	for _, op := range ops {
		if op == OpConv3x3 {
			hasConv3x3 = true
		}
		if op == OpIdentity {
			hasIdentity = true
		}
	}
	if !hasConv3x3 {
		t.Error("AllOperations() should include OpConv3x3")
	}
	if !hasIdentity {
		t.Error("AllOperations() should include OpIdentity")
	}
}

func TestDefaultOperations(t *testing.T) {
	ops := DefaultOperations()

	if len(ops) == 0 {
		t.Fatal("DefaultOperations() returned empty slice")
	}

	// Default should include identity and zero
	hasIdentity := false
	hasZero := false
	for _, op := range ops {
		if op == OpIdentity {
			hasIdentity = true
		}
		if op == OpZero {
			hasZero = true
		}
	}
	if !hasIdentity {
		t.Error("DefaultOperations() should include OpIdentity")
	}
	if !hasZero {
		t.Error("DefaultOperations() should include OpZero")
	}
}

// ─── Cell Tests ─────────────────────────────────────────────────────────────

func TestNewCell(t *testing.T) {
	cell := NewCell(NormalCell, 4, 2, 2)

	if cell.Type != NormalCell {
		t.Errorf("cell.Type = %v, want NormalCell", cell.Type)
	}
	if len(cell.Nodes) != 4 {
		t.Errorf("len(cell.Nodes) = %d, want 4", len(cell.Nodes))
	}
	if cell.NumInputNodes != 2 {
		t.Errorf("cell.NumInputNodes = %d, want 2", cell.NumInputNodes)
	}

	for i, node := range cell.Nodes {
		if len(node.Edges) != 2 {
			t.Errorf("node[%d] has %d edges, want 2", i, len(node.Edges))
		}
	}
}

func TestCellTotalNodes(t *testing.T) {
	cell := NewCell(NormalCell, 4, 2, 2)
	if got := cell.TotalNodes(); got != 6 {
		t.Errorf("TotalNodes() = %d, want 6", got)
	}
}

func TestCellNodeIndex(t *testing.T) {
	cell := NewCell(NormalCell, 4, 2, 2)

	tests := []struct {
		intermediate int
		want         int
	}{
		{0, 2},
		{1, 3},
		{2, 4},
		{3, 5},
	}

	for _, tt := range tests {
		if got := cell.NodeIndex(tt.intermediate); got != tt.want {
			t.Errorf("NodeIndex(%d) = %d, want %d", tt.intermediate, got, tt.want)
		}
	}
}

func TestCellValidInputsForNode(t *testing.T) {
	cell := NewCell(NormalCell, 4, 2, 2)

	// Node at global index 2 (first intermediate) can connect to 0,1
	inputs := cell.ValidInputsForNode(2)
	if len(inputs) != 2 {
		t.Fatalf("ValidInputsForNode(2) returned %d inputs, want 2", len(inputs))
	}
	if inputs[0] != 0 || inputs[1] != 1 {
		t.Errorf("ValidInputsForNode(2) = %v, want [0 1]", inputs)
	}

	// Node at global index 5 (last intermediate) can connect to 0,1,2,3,4
	inputs5 := cell.ValidInputsForNode(5)
	if len(inputs5) != 5 {
		t.Fatalf("ValidInputsForNode(5) returned %d inputs, want 5", len(inputs5))
	}
}

func TestCellClone(t *testing.T) {
	cell := NewCell(NormalCell, 4, 2, 2)
	cell.Nodes[0].Edges[0] = Edge{InputNode: 1, Operation: OpConv3x3}

	clone := cell.Clone()

	// Verify clone matches
	if clone.Nodes[0].Edges[0].Operation != OpConv3x3 {
		t.Error("clone should have same operation as original")
	}

	// Modify clone should not affect original
	clone.Nodes[0].Edges[0].Operation = OpConv5x5
	if cell.Nodes[0].Edges[0].Operation != OpConv3x3 {
		t.Error("modifying clone should not affect original")
	}
}

func TestCellHash(t *testing.T) {
	cell1 := NewCell(NormalCell, 4, 2, 2)
	cell2 := NewCell(NormalCell, 4, 2, 2)

	// Same structure should have same hash
	if cell1.Hash() != cell2.Hash() {
		t.Error("identical cells should have same hash")
	}

	// Different structure should have different hash
	cell2.Nodes[0].Edges[0].Operation = OpConv3x3
	if cell1.Hash() == cell2.Hash() {
		t.Error("different cells should have different hash")
	}
}

func TestCellIsValid(t *testing.T) {
	cell := NewCell(NormalCell, 4, 2, 2)
	// Default cell has OpNone, which is at index 0
	// Edges default to InputNode=0, which is valid for all nodes
	if !cell.IsValid() {
		t.Error("default cell should be valid")
	}

	// Set valid edges
	cell.Nodes[0].Edges[0] = Edge{InputNode: 0, Operation: OpConv3x3}
	cell.Nodes[0].Edges[1] = Edge{InputNode: 1, Operation: OpIdentity}
	if !cell.IsValid() {
		t.Error("cell with valid edges should be valid")
	}

	// Invalid: node 0 (global idx 2) connecting to node 2 (which is itself)
	cell.Nodes[0].Edges[0] = Edge{InputNode: 2, Operation: OpConv3x3}
	if cell.IsValid() {
		t.Error("cell with self-referencing edge should be invalid")
	}
}

func TestCellParameterEstimate(t *testing.T) {
	cell := NewCell(NormalCell, 4, 2, 2)

	// All OpNone = 0 params
	if got := cell.ParameterEstimate(); got != 0 {
		t.Errorf("empty cell ParameterEstimate() = %d, want 0", got)
	}

	// Set some operations
	cell.Nodes[0].Edges[0].Operation = OpConv3x3  // 9
	cell.Nodes[0].Edges[1].Operation = OpConv5x5  // 25
	cell.Nodes[1].Edges[0].Operation = OpIdentity // 0

	expected := 9 + 25 + 0
	if got := cell.ParameterEstimate(); got != expected {
		t.Errorf("ParameterEstimate() = %d, want %d", got, expected)
	}
}

func TestCellUsedOperations(t *testing.T) {
	cell := NewCell(NormalCell, 2, 2, 2)
	cell.Nodes[0].Edges[0] = Edge{InputNode: 0, Operation: OpConv3x3}
	cell.Nodes[0].Edges[1] = Edge{InputNode: 1, Operation: OpConv3x3}
	cell.Nodes[1].Edges[0] = Edge{InputNode: 0, Operation: OpIdentity}
	cell.Nodes[1].Edges[1] = Edge{InputNode: 1, Operation: OpMaxPool3x3}

	ops := cell.UsedOperations()

	// Should have 3 unique operations
	if len(ops) != 3 {
		t.Errorf("UsedOperations() returned %d ops, want 3", len(ops))
	}
}

func TestCellTypeString(t *testing.T) {
	if got := NormalCell.String(); got != "normal" {
		t.Errorf("NormalCell.String() = %q, want %q", got, "normal")
	}
	if got := ReductionCell.String(); got != "reduction" {
		t.Errorf("ReductionCell.String() = %q, want %q", got, "reduction")
	}
}

// ─── Architecture Tests ─────────────────────────────────────────────────────

func TestNewArchitecture(t *testing.T) {
	normal := NewCell(NormalCell, 4, 2, 2)
	reduction := NewCell(ReductionCell, 4, 2, 2)
	arch := NewArchitecture(normal, reduction)

	if arch.ID == "" {
		t.Error("architecture should have non-empty ID")
	}
	if arch.NormalCell == nil {
		t.Error("architecture should have normal cell")
	}
	if arch.ReductionCell == nil {
		t.Error("architecture should have reduction cell")
	}
	if arch.Metadata.CreatedAt.IsZero() {
		t.Error("architecture should have creation time")
	}
}

func TestArchitectureClone(t *testing.T) {
	normal := NewCell(NormalCell, 4, 2, 2)
	normal.Nodes[0].Edges[0] = Edge{InputNode: 0, Operation: OpConv3x3}
	reduction := NewCell(ReductionCell, 4, 2, 2)
	arch := NewArchitecture(normal, reduction)

	clone := arch.Clone()

	// Clone should have different ID
	if clone.ID == arch.ID {
		t.Error("clone should have different ID")
	}

	// Clone should have parent ID set
	if clone.Metadata.ParentID != arch.ID {
		t.Error("clone's parent ID should be original's ID")
	}

	// Modifying clone should not affect original
	clone.NormalCell.Nodes[0].Edges[0].Operation = OpConv5x5
	if arch.NormalCell.Nodes[0].Edges[0].Operation != OpConv3x3 {
		t.Error("modifying clone should not affect original")
	}
}

func TestArchitectureHash(t *testing.T) {
	normal1 := NewCell(NormalCell, 4, 2, 2)
	reduction1 := NewCell(ReductionCell, 4, 2, 2)
	arch1 := NewArchitecture(normal1, reduction1)

	normal2 := NewCell(NormalCell, 4, 2, 2)
	reduction2 := NewCell(ReductionCell, 4, 2, 2)
	arch2 := NewArchitecture(normal2, reduction2)

	// Same structure should have same hash
	if arch1.Hash() != arch2.Hash() {
		t.Error("identical architectures should have same hash")
	}

	// Different structure should have different hash
	normal2.Nodes[0].Edges[0].Operation = OpConv3x3
	arch3 := NewArchitecture(normal2, reduction2)
	if arch1.Hash() == arch3.Hash() {
		t.Error("different architectures should have different hash")
	}
}

func TestArchitectureIsValid(t *testing.T) {
	normal := NewCell(NormalCell, 4, 2, 2)
	reduction := NewCell(ReductionCell, 4, 2, 2)
	arch := NewArchitecture(normal, reduction)

	if !arch.IsValid() {
		t.Error("architecture with valid cells should be valid")
	}

	// Nil cell should be invalid
	arch.NormalCell = nil
	if arch.IsValid() {
		t.Error("architecture with nil normal cell should be invalid")
	}
}

func TestArchitectureJSON(t *testing.T) {
	normal := NewCell(NormalCell, 4, 2, 2)
	normal.Nodes[0].Edges[0] = Edge{InputNode: 0, Operation: OpConv3x3}
	reduction := NewCell(ReductionCell, 4, 2, 2)
	arch := NewArchitecture(normal, reduction)
	arch.Metadata.Fitness = 0.95

	// Serialize
	data, err := arch.ToJSON(true)
	if err != nil {
		t.Fatalf("ToJSON() error: %v", err)
	}

	// Deserialize
	parsed, err := ArchitectureFromJSON(data)
	if err != nil {
		t.Fatalf("ArchitectureFromJSON() error: %v", err)
	}

	if parsed.ID != arch.ID {
		t.Errorf("parsed ID = %q, want %q", parsed.ID, arch.ID)
	}
	if parsed.NormalCell.Nodes[0].Edges[0].Operation != OpConv3x3 {
		t.Error("parsed architecture should preserve operations")
	}
}

func TestArchitectureGenotype(t *testing.T) {
	normal := NewCell(NormalCell, 4, 2, 2)
	normal.Nodes[0].Edges[0] = Edge{InputNode: 0, Operation: OpConv3x3}
	normal.Nodes[0].Edges[1] = Edge{InputNode: 1, Operation: OpIdentity}
	reduction := NewCell(ReductionCell, 4, 2, 2)
	arch := NewArchitecture(normal, reduction)

	// Convert to genotype
	normalGeno, reductionGeno := arch.ToGenotype()

	// Expected length: 4 nodes × 2 edges × 2 values = 16
	if len(normalGeno) != 16 {
		t.Fatalf("normal genotype length = %d, want 16", len(normalGeno))
	}
	if len(reductionGeno) != 16 {
		t.Fatalf("reduction genotype length = %d, want 16", len(reductionGeno))
	}

	// First edge: input=0, op=OpConv3x3(3)
	if normalGeno[0] != 0 || normalGeno[1] != int(OpConv3x3) {
		t.Errorf("genotype[0:2] = [%d, %d], want [0, %d]", normalGeno[0], normalGeno[1], OpConv3x3)
	}

	// Reconstruct from genotype
	reconstructed, err := FromGenotype(normalGeno, reductionGeno, 4, 2, 2)
	if err != nil {
		t.Fatalf("FromGenotype() error: %v", err)
	}

	if reconstructed.NormalCell.Nodes[0].Edges[0].Operation != OpConv3x3 {
		t.Error("reconstructed architecture should preserve operations")
	}
}

// ─── SearchSpace Tests ──────────────────────────────────────────────────────

func TestDefaultSearchSpace(t *testing.T) {
	space := DefaultSearchSpace()

	if space.NumNodes != 4 {
		t.Errorf("NumNodes = %d, want 4", space.NumNodes)
	}
	if space.NumInputNodes != 2 {
		t.Errorf("NumInputNodes = %d, want 2", space.NumInputNodes)
	}
	if space.EdgesPerNode != 2 {
		t.Errorf("EdgesPerNode = %d, want 2", space.EdgesPerNode)
	}
	if len(space.Operations) == 0 {
		t.Error("Operations should not be empty")
	}
}

func TestNewSearchSpace(t *testing.T) {
	ops := []OperationType{OpConv3x3, OpIdentity, OpZero}
	space, err := NewSearchSpace(ops, 3, 2, 2, 42)
	if err != nil {
		t.Fatalf("NewSearchSpace() error: %v", err)
	}

	if len(space.Operations) != 3 {
		t.Errorf("len(Operations) = %d, want 3", len(space.Operations))
	}
}

func TestNewSearchSpaceValidation(t *testing.T) {
	// Empty operations
	_, err := NewSearchSpace(nil, 3, 2, 2, 42)
	if err == nil {
		t.Error("expected error for empty operations")
	}

	// Invalid numNodes
	ops := []OperationType{OpConv3x3}
	_, err = NewSearchSpace(ops, 0, 2, 2, 42)
	if err == nil {
		t.Error("expected error for numNodes=0")
	}
}

func TestSearchSpaceSize(t *testing.T) {
	space := DefaultSearchSpace()
	size := space.Size()

	// Size should be positive and large
	if size <= 0 {
		t.Errorf("Size() = %f, want > 0", size)
	}
	// With 9 ops, 4 nodes, 2 edges: should be very large
	if size < 1e6 {
		t.Errorf("Size() = %f, expected > 1e6", size)
	}
}

func TestSampleRandomArchitecture(t *testing.T) {
	space := DefaultSearchSpace()
	space.SetSeed(42)

	arch := space.SampleRandomArchitecture()

	if arch == nil {
		t.Fatal("SampleRandomArchitecture() returned nil")
	}
	if arch.ID == "" {
		t.Error("sampled architecture should have ID")
	}
	if !arch.IsValid() {
		t.Error("sampled architecture should be valid")
	}

	// Verify operations are from the search space
	if err := space.Validate(arch); err != nil {
		t.Errorf("sampled architecture failed validation: %v", err)
	}
}

func TestSampleReproducibility(t *testing.T) {
	space1 := DefaultSearchSpace()
	space1.SetSeed(42)
	arch1 := space1.SampleRandomArchitecture()

	space2 := DefaultSearchSpace()
	space2.SetSeed(42)
	arch2 := space2.SampleRandomArchitecture()

	if arch1.Hash() != arch2.Hash() {
		t.Error("same seed should produce same architecture")
	}
}

func TestMutate(t *testing.T) {
	space := DefaultSearchSpace()
	space.SetSeed(42)

	parent := space.SampleRandomArchitecture()
	child := space.Mutate(parent)

	// Child should be different from parent
	if child.ID == parent.ID {
		t.Error("child should have different ID")
	}

	// Child should reference parent
	if child.Metadata.ParentID != parent.ID {
		t.Error("child should reference parent")
	}

	// Child should be valid
	if !child.IsValid() {
		t.Error("mutated child should be valid")
	}

	// Child should have mutation type set
	if child.Metadata.MutationType == "" {
		t.Error("child should have mutation type")
	}
}

func TestCrossover(t *testing.T) {
	space := DefaultSearchSpace()
	space.SetSeed(42)

	parent1 := space.SampleRandomArchitecture()
	parent2 := space.SampleRandomArchitecture()
	child := space.Crossover(parent1, parent2)

	if child == nil {
		t.Fatal("Crossover() returned nil")
	}
	if child.Metadata.MutationType != "crossover" {
		t.Errorf("mutation type = %q, want %q", child.Metadata.MutationType, "crossover")
	}
}

func TestPopulateInitial(t *testing.T) {
	space := DefaultSearchSpace()
	space.SetSeed(42)

	pop := space.PopulateInitial(10)

	if len(pop) != 10 {
		t.Fatalf("PopulateInitial(10) returned %d, want 10", len(pop))
	}

	for i, arch := range pop {
		if arch == nil {
			t.Errorf("population[%d] is nil", i)
			continue
		}
		if !arch.IsValid() {
			t.Errorf("population[%d] is invalid", i)
		}
		if arch.Metadata.Generation != 0 {
			t.Errorf("population[%d] generation = %d, want 0", i, arch.Metadata.Generation)
		}
	}
}

func TestSearchSpaceValidate(t *testing.T) {
	space := DefaultSearchSpace()
	space.SetSeed(42)

	// Valid architecture from this space
	arch := space.SampleRandomArchitecture()
	if err := space.Validate(arch); err != nil {
		t.Errorf("valid architecture failed validation: %v", err)
	}

	// Architecture with invalid operation
	arch.NormalCell.Nodes[0].Edges[0].Operation = OpConv7x7 // Not in default ops
	if err := space.Validate(arch); err == nil {
		t.Error("expected error for architecture with disallowed operation")
	}
}

func TestSearchSpaceString(t *testing.T) {
	space := DefaultSearchSpace()
	s := space.String()

	if s == "" {
		t.Error("String() should not be empty")
	}
}
