// Package searchspace defines the search space for Neural Architecture Search.
// It contains types for representing operations, cells, and full architectures.
//
// The design follows the NASNet/DARTS convention of cell-based search spaces,
// where we search for a small repeatable "cell" that gets stacked to form
// the full network.
//
// References:
// - NASNet: https://arxiv.org/abs/1707.07012
// - DARTS: https://arxiv.org/abs/1806.09055
// - Regularized Evolution: https://arxiv.org/abs/1802.01548
package searchspace

import (
	"fmt"
	"strings"
)

// OperationType represents the type of operation in a neural network cell.
// Using a custom type (not raw int) provides type safety - the compiler will
// catch if you accidentally pass a regular int where an OperationType is expected.
//
// Go doesn't have enums like Python or Java, so we use iota with a custom type.
// This is the idiomatic Go pattern for creating enum-like constants.
type OperationType int

// Operation type constants using iota.
// iota starts at 0 and auto-increments for each constant in the block.
// We use iota because:
// 1. It's less error-prone than manually assigning numbers
// 2. Adding new operations is easy (just add a new line)
// 3. The values are guaranteed to be unique
//
// IMPORTANT: Never reorder these constants in production code!
// The numeric values are used for serialization/storage. Reordering
// would break existing saved architectures.
const (
	// OpNone represents no operation (used for padding or invalid states)
	// Value: 0 (iota starts at 0)
	OpNone OperationType = iota

	// OpIdentity is a skip connection (identity mapping).
	// Passes input unchanged. Essential for residual learning.
	// Value: 1
	OpIdentity

	// OpConv1x1 applies a 1x1 convolution (pointwise convolution).
	// Used to change channel dimensions without spatial mixing.
	// Parameters: channels_in × channels_out
	// Value: 2
	OpConv1x1

	// OpConv3x3 applies a 3x3 convolution with padding=1 to preserve dimensions.
	// The workhorse of CNNs. 3x3 is the most common kernel size because:
	// - It's the smallest size that can capture left/right, up/down, center
	// - Very efficient on modern GPUs
	// Parameters: channels × 9 × channels
	// Value: 3
	OpConv3x3

	// OpConv5x5 applies a 5x5 convolution with padding=2.
	// Larger receptive field than 3x3, but more parameters (25 vs 9).
	// Value: 4
	OpConv5x5

	// OpConv7x7 applies a 7x7 convolution with padding=3.
	// Even larger receptive field. Often used in first layer of networks.
	// Value: 5
	OpConv7x7

	// OpSepConv3x3 is a depthwise separable 3x3 convolution.
	// Factorizes regular convolution into:
	// 1. Depthwise conv: separate 3x3 filter per channel
	// 2. Pointwise conv: 1x1 conv to mix channels
	// Much fewer parameters than regular conv: ~1/9th for 3x3
	// Key to efficient architectures like MobileNet.
	// Value: 6
	OpSepConv3x3

	// OpSepConv5x5 is a depthwise separable 5x5 convolution.
	// Value: 7
	OpSepConv5x5

	// OpDilConv3x3 is a dilated (atrous) 3x3 convolution with dilation=2.
	// Increases receptive field without adding parameters.
	// Receptive field becomes 5x5 with only 9 weights.
	// Value: 8
	OpDilConv3x3

	// OpDilConv5x5 is a dilated 5x5 convolution with dilation=2.
	// Value: 9
	OpDilConv5x5

	// OpMaxPool3x3 applies 3x3 max pooling with stride=1, padding=1.
	// Preserves spatial dimensions. Provides translation invariance.
	// Value: 10
	OpMaxPool3x3

	// OpAvgPool3x3 applies 3x3 average pooling with stride=1, padding=1.
	// Smoother than max pooling. Often used in reduction cells.
	// Value: 11
	OpAvgPool3x3

	// OpZero represents a "zero" operation - no connection.
	// Output is always zeros. Used to represent sparse connections
	// in the search space (i.e., "no edge here").
	// Value: 12
	OpZero

	// opCount is a sentinel value to count operations.
	// Not exported (lowercase) because it's only for internal use.
	// Value: 13 (number of operations above)
	opCount
)

// operationNames maps operation types to human-readable names.
// Using a fixed-size array instead of map for performance.
// Index corresponds to OperationType value.
var operationNames = [opCount]string{
	OpNone:       "none",
	OpIdentity:   "identity",
	OpConv1x1:    "conv_1x1",
	OpConv3x3:    "conv_3x3",
	OpConv5x5:    "conv_5x5",
	OpConv7x7:    "conv_7x7",
	OpSepConv3x3: "sep_conv_3x3",
	OpSepConv5x5: "sep_conv_5x5",
	OpDilConv3x3: "dil_conv_3x3",
	OpDilConv5x5: "dil_conv_5x5",
	OpMaxPool3x3: "max_pool_3x3",
	OpAvgPool3x3: "avg_pool_3x3",
	OpZero:       "zero",
}

// String returns the human-readable name of the operation.
// Implementing String() makes OperationType satisfy fmt.Stringer,
// so it prints nicely with fmt.Printf("%v", op).
//
// Example:
//
//	op := OpConv3x3
//	fmt.Println(op) // Prints: conv_3x3
func (o OperationType) String() string {
	if o < 0 || o >= opCount {
		return fmt.Sprintf("unknown(%d)", o)
	}
	return operationNames[o]
}

// IsValid returns true if the operation type is a valid, usable operation.
// OpNone is considered invalid as it represents an uninitialized state.
func (o OperationType) IsValid() bool {
	return o > OpNone && o < opCount
}

// IsPooling returns true if this is a pooling operation.
// Useful for architecture analysis and when applying stride.
func (o OperationType) IsPooling() bool {
	return o == OpMaxPool3x3 || o == OpAvgPool3x3
}

// IsConvolution returns true if this is any convolution operation.
func (o OperationType) IsConvolution() bool {
	switch o {
	case OpConv1x1, OpConv3x3, OpConv5x5, OpConv7x7,
		OpSepConv3x3, OpSepConv5x5, OpDilConv3x3, OpDilConv5x5:
		return true
	default:
		return false
	}
}

// ParameterCount returns the relative parameter count for this operation.
// Values are normalized relative to a standard 3x3 conv (value = 9).
// This is used for quick architecture size estimation.
//
// Returns:
//   - The relative parameter count as an integer
//   - 0 for operations with no learnable parameters
func (o OperationType) ParameterCount() int {
	switch o {
	case OpNone, OpZero:
		return 0
	case OpIdentity:
		return 0 // No parameters
	case OpConv1x1:
		return 1 // 1x1 kernel
	case OpConv3x3:
		return 9 // 3x3 kernel
	case OpConv5x5:
		return 25 // 5x5 kernel
	case OpConv7x7:
		return 49 // 7x7 kernel
	case OpSepConv3x3:
		return 12 // 3x3 depthwise + 1x1 pointwise ≈ 9 + 1 × channels
	case OpSepConv5x5:
		return 30 // 5x5 depthwise + 1x1 pointwise
	case OpDilConv3x3, OpDilConv5x5:
		return 9 // Same params as regular conv
	case OpMaxPool3x3, OpAvgPool3x3:
		return 0 // No learnable parameters
	default:
		return 0
	}
}

// ParseOperationType converts a string to an OperationType.
// This is useful when loading architectures from JSON/YAML.
//
// Parameters:
//   - s: the string name of the operation (case-insensitive)
//
// Returns:
//   - The corresponding OperationType
//   - An error if the string doesn't match any known operation
//
// Example:
//
//	op, err := ParseOperationType("conv_3x3")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	// op == OpConv3x3
func ParseOperationType(s string) (OperationType, error) {
	lower := strings.ToLower(strings.TrimSpace(s))
	for i := OperationType(0); i < opCount; i++ {
		if operationNames[i] == lower {
			return i, nil
		}
	}
	return OpNone, fmt.Errorf("unknown operation type: %q", s)
}

// AllOperations returns a slice of all valid operation types.
// Useful for iterating over all possible operations in search algorithms.
//
// Note: This excludes OpNone as it's not a valid operation choice.
func AllOperations() []OperationType {
	ops := make([]OperationType, 0, opCount-2) // Exclude OpNone and OpZero typically
	for i := OpIdentity; i < opCount; i++ {
		ops = append(ops, i)
	}
	return ops
}

// DefaultOperations returns the standard operation set used in NAS benchmarks.
// This matches the DARTS search space operations.
//
// Reference: DARTS paper, Section 3.1
func DefaultOperations() []OperationType {
	return []OperationType{
		OpIdentity,
		OpConv3x3,
		OpConv5x5,
		OpSepConv3x3,
		OpSepConv5x5,
		OpDilConv3x3,
		OpMaxPool3x3,
		OpAvgPool3x3,
		OpZero, // Allows the search to "turn off" connections
	}
}
