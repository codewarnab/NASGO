package utils

import (
	"context"
	"io"
	"log/slog"
	"os"
	"strings"
)

// Logger provides structured logging for the NAS system.
// Uses Go's built-in slog package (Go 1.21+).
//
// Key features:
// - Structured key-value logging
// - Child loggers with context (Logger.With)
// - JSON or text output
// - Level-based filtering
//
// Reference: Go slog best practices 2024-2025
type Logger struct {
	*slog.Logger
}

// NewLogger creates a logger from the logging config.
func NewLogger(cfg LoggingConfig) (*Logger, error) {
	// Determine output writer
	var writer io.Writer = os.Stdout
	if cfg.File != "" {
		f, err := os.OpenFile(cfg.File, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return nil, err
		}
		writer = f
	}

	// Parse log level
	var level slog.Level
	switch strings.ToLower(cfg.Level) {
	case "debug":
		level = slog.LevelDebug
	case "info":
		level = slog.LevelInfo
	case "warn", "warning":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	default:
		level = slog.LevelInfo
	}

	// Build handler options
	opts := &slog.HandlerOptions{
		Level:     level,
		AddSource: cfg.IncludeSource,
	}

	// Create handler based on format
	var handler slog.Handler
	switch strings.ToLower(cfg.Format) {
	case "json":
		handler = slog.NewJSONHandler(writer, opts)
	default:
		handler = slog.NewTextHandler(writer, opts)
	}

	return &Logger{slog.New(handler)}, nil
}

// With creates a child logger with additional context.
// Use this to add persistent context like request IDs.
//
// Example:
//
//	logger := baseLogger.With("experiment_id", exp.ID)
//	logger.Info("starting search")  // includes experiment_id
func (l *Logger) With(args ...any) *Logger {
	return &Logger{l.Logger.With(args...)}
}

// WithContext adds common context fields.
// Useful for request-scoped logging.
func (l *Logger) WithContext(ctx context.Context) *Logger {
	// Extract values from context if present
	// (Implementation depends on what you store in context)
	_ = ctx
	return l
}

// SearchLogger creates a logger for search operations.
func (l *Logger) SearchLogger(strategy, experimentID string) *Logger {
	return l.With(
		"component", "search",
		"strategy", strategy,
		"experiment_id", experimentID,
	)
}

// EvaluationLogger creates a logger for evaluation operations.
func (l *Logger) EvaluationLogger(evaluatorType string) *Logger {
	return l.With(
		"component", "evaluator",
		"type", evaluatorType,
	)
}

// Progress logs evaluation progress at info level.
func (l *Logger) Progress(current, total int, bestFitness float64) {
	percent := float64(current) / float64(total) * 100
	l.Info("search progress",
		slog.Int("evaluation", current),
		slog.Int("total", total),
		slog.Float64("percent", percent),
		slog.Float64("best_fitness", bestFitness),
	)
}

// Architecture logs an evaluated architecture.
func (l *Logger) Architecture(archID string, fitness float64, generation int) {
	l.Debug("architecture evaluated",
		slog.String("arch_id", archID[:8]),
		slog.Float64("fitness", fitness),
		slog.Int("generation", generation),
	)
}
