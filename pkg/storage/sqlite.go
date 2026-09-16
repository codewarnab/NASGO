// Package storage provides persistent storage for NAS experiments.
package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	_ "modernc.org/sqlite" // Pure Go SQLite driver

	"nas-go/pkg/searchspace"
)

// SQLiteStorage stores experiments and architectures in SQLite.
// Uses modernc.org/sqlite - a pure Go driver (no CGO required).
//
// Schema:
// - experiments: metadata about each search run
// - architectures: all evaluated architectures
// - checkpoints: periodic snapshots for resumption
//
// Why SQLite?
// - Zero configuration (just a file)
// - Portable across platforms
// - Good enough for typical NAS experiments
// - Easy to query with standard SQL tools
type SQLiteStorage struct {
	db   *sql.DB
	path string
}

// NewSQLiteStorage creates or opens a SQLite database.
func NewSQLiteStorage(path string) (*SQLiteStorage, error) {
	// Open database (creates if not exists)
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("opening database: %w", err)
	}

	// Configure for concurrent access
	// SQLite is single-writer, so we limit write connections
	db.SetMaxOpenConns(1)

	storage := &SQLiteStorage{db: db, path: path}

	// Create tables if they don't exist
	if err := storage.initSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("initializing schema: %w", err)
	}

	return storage, nil
}

// initSchema creates tables if they don't exist.
func (s *SQLiteStorage) initSchema() error {
	schema := `
	-- Experiments table
	CREATE TABLE IF NOT EXISTS experiments (
		id TEXT PRIMARY KEY,
		name TEXT NOT NULL,
		description TEXT,
		config_json TEXT NOT NULL,
		strategy TEXT NOT NULL,
		status TEXT NOT NULL DEFAULT 'running',
		started_at DATETIME NOT NULL,
		completed_at DATETIME,
		best_fitness REAL,
		best_arch_id TEXT,
		total_evaluations INTEGER DEFAULT 0
	);

	-- Architectures table
	CREATE TABLE IF NOT EXISTS architectures (
		id TEXT PRIMARY KEY,
		experiment_id TEXT NOT NULL,
		arch_json TEXT NOT NULL,
		fitness REAL,
		accuracy REAL,
		parameters INTEGER,
		generation INTEGER,
		parent_id TEXT,
		mutation_type TEXT,
		evaluated_at DATETIME NOT NULL,
		evaluation_time_ms INTEGER,
		FOREIGN KEY (experiment_id) REFERENCES experiments(id)
	);

	-- Checkpoints table
	CREATE TABLE IF NOT EXISTS checkpoints (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		experiment_id TEXT NOT NULL,
		evaluation_number INTEGER NOT NULL,
		population_json TEXT,
		best_fitness REAL,
		created_at DATETIME NOT NULL,
		FOREIGN KEY (experiment_id) REFERENCES experiments(id)
	);

	-- Indexes for common queries
	CREATE INDEX IF NOT EXISTS idx_arch_experiment ON architectures(experiment_id);
	CREATE INDEX IF NOT EXISTS idx_arch_fitness ON architectures(fitness DESC);
	CREATE INDEX IF NOT EXISTS idx_checkpoint_experiment ON checkpoints(experiment_id);
	`

	_, err := s.db.Exec(schema)
	return err
}

// Experiment represents a stored experiment.
type Experiment struct {
	ID               string
	Name             string
	Description      string
	ConfigJSON       string
	Strategy         string
	Status           string
	StartedAt        time.Time
	CompletedAt      *time.Time
	BestFitness      *float64
	BestArchID       *string
	TotalEvaluations int
}

// CreateExperiment stores a new experiment.
func (s *SQLiteStorage) CreateExperiment(ctx context.Context, exp Experiment) error {
	query := `
		INSERT INTO experiments (id, name, description, config_json, strategy, status, started_at)
		VALUES (?, ?, ?, ?, ?, 'running', ?)
	`
	_, err := s.db.ExecContext(ctx, query,
		exp.ID, exp.Name, exp.Description, exp.ConfigJSON, exp.Strategy, exp.StartedAt,
	)
	return err
}

// UpdateExperiment updates experiment status and results.
func (s *SQLiteStorage) UpdateExperiment(ctx context.Context, exp Experiment) error {
	query := `
		UPDATE experiments
		SET status = ?, completed_at = ?, best_fitness = ?, best_arch_id = ?, total_evaluations = ?
		WHERE id = ?
	`
	_, err := s.db.ExecContext(ctx, query,
		exp.Status, exp.CompletedAt, exp.BestFitness, exp.BestArchID, exp.TotalEvaluations, exp.ID,
	)
	return err
}

// GetExperiment retrieves an experiment by ID.
func (s *SQLiteStorage) GetExperiment(ctx context.Context, id string) (*Experiment, error) {
	query := `SELECT id, name, description, config_json, strategy, status, 
	          started_at, completed_at, best_fitness, best_arch_id, total_evaluations
	          FROM experiments WHERE id = ?`

	exp := &Experiment{}
	err := s.db.QueryRowContext(ctx, query, id).Scan(
		&exp.ID, &exp.Name, &exp.Description, &exp.ConfigJSON, &exp.Strategy, &exp.Status,
		&exp.StartedAt, &exp.CompletedAt, &exp.BestFitness, &exp.BestArchID, &exp.TotalEvaluations,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return exp, err
}

// SaveArchitecture stores an evaluated architecture.
func (s *SQLiteStorage) SaveArchitecture(ctx context.Context, experimentID string, arch *searchspace.Architecture) error {
	archJSON, err := arch.ToJSON(false)
	if err != nil {
		return err
	}

	query := `
		INSERT INTO architectures 
		(id, experiment_id, arch_json, fitness, parameters, generation, parent_id, mutation_type, evaluated_at, evaluation_time_ms)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`
	_, err = s.db.ExecContext(ctx, query,
		arch.ID,
		experimentID,
		string(archJSON),
		arch.Metadata.Fitness,
		arch.ParameterEstimate(),
		arch.Metadata.Generation,
		nilIfEmpty(arch.Metadata.ParentID),
		nilIfEmpty(arch.Metadata.MutationType),
		arch.Metadata.CreatedAt,
		arch.Metadata.EvaluationTime.Milliseconds(),
	)
	return err
}

// GetTopArchitectures returns the best architectures for an experiment.
func (s *SQLiteStorage) GetTopArchitectures(ctx context.Context, experimentID string, limit int) ([]*searchspace.Architecture, error) {
	query := `
		SELECT arch_json FROM architectures
		WHERE experiment_id = ?
		ORDER BY fitness DESC
		LIMIT ?
	`

	rows, err := s.db.QueryContext(ctx, query, experimentID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var archs []*searchspace.Architecture
	for rows.Next() {
		var archJSON string
		if err := rows.Scan(&archJSON); err != nil {
			return nil, err
		}
		arch, err := searchspace.ArchitectureFromJSON([]byte(archJSON))
		if err != nil {
			continue // Skip invalid entries
		}
		archs = append(archs, arch)
	}

	return archs, rows.Err()
}

// SaveCheckpoint saves a search checkpoint.
func (s *SQLiteStorage) SaveCheckpoint(ctx context.Context, experimentID string, evalNum int, population []*searchspace.Architecture, bestFitness float64) error {
	// Serialize population
	popData, err := json.Marshal(population)
	if err != nil {
		return err
	}

	query := `
		INSERT INTO checkpoints (experiment_id, evaluation_number, population_json, best_fitness, created_at)
		VALUES (?, ?, ?, ?, ?)
	`
	_, err = s.db.ExecContext(ctx, query, experimentID, evalNum, string(popData), bestFitness, time.Now())
	return err
}

// Checkpoint is a versioned resumable search snapshot.
type Checkpoint struct {
	Version          int                         `json:"version"`
	Strategy         string                      `json:"strategy"`
	EvaluationNumber int                         `json:"evaluation_number"`
	History          []*searchspace.Architecture `json:"history"`
	BestFitness      float64                     `json:"best_fitness"`
	Population       []*searchspace.Architecture `json:"population,omitempty"`
	StrategyRNG      uint64                      `json:"strategy_rng,omitempty"`
	SearchSpaceRNG   uint64                      `json:"search_space_rng,omitempty"`
	ConfigJSON       string                      `json:"config_json,omitempty"`
}

// SaveSearchCheckpoint stores a complete versioned snapshot.
func (s *SQLiteStorage) SaveSearchCheckpoint(ctx context.Context, experimentID string, cp Checkpoint) error {
	if cp.Version == 0 {
		cp.Version = 1
	}
	data, err := json.Marshal(cp)
	if err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, `INSERT INTO checkpoints (experiment_id,evaluation_number,population_json,best_fitness,created_at) VALUES (?,?,?,?,?)`, experimentID, cp.EvaluationNumber, string(data), cp.BestFitness, time.Now())
	return err
}

// LoadLatestCheckpoint restores the latest snapshot for an experiment.
func (s *SQLiteStorage) LoadLatestCheckpoint(ctx context.Context, experimentID string) (*Checkpoint, error) {
	var raw string
	err := s.db.QueryRowContext(ctx, `SELECT population_json FROM checkpoints WHERE experiment_id=? ORDER BY evaluation_number DESC,id DESC LIMIT 1`, experimentID).Scan(&raw)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var cp Checkpoint
	if err = json.Unmarshal([]byte(raw), &cp); err != nil {
		return nil, fmt.Errorf("decoding checkpoint: %w", err)
	}
	if cp.Version != 1 {
		return nil, fmt.Errorf("unsupported checkpoint version %d", cp.Version)
	}
	if cp.Strategy != "random" && cp.Strategy != "evolutionary" && cp.Strategy != "regularized" {
		return nil, fmt.Errorf("invalid checkpoint strategy %q", cp.Strategy)
	}
	if cp.EvaluationNumber < 0 || cp.EvaluationNumber != len(cp.History) {
		return nil, fmt.Errorf("inconsistent checkpoint evaluation number %d for history length %d", cp.EvaluationNumber, len(cp.History))
	}
	if cp.EvaluationNumber > 0 && cp.SearchSpaceRNG == 0 {
		return nil, fmt.Errorf("checkpoint is missing search-space RNG state")
	}
	if (cp.Strategy == "evolutionary" || cp.Strategy == "regularized") && cp.EvaluationNumber > 0 {
		if len(cp.Population) == 0 || cp.StrategyRNG == 0 {
			return nil, fmt.Errorf("checkpoint is missing evolutionary population or RNG state")
		}
	}
	return &cp, nil
}

// Close closes the database connection.
func (s *SQLiteStorage) Close() error {
	return s.db.Close()
}

func nilIfEmpty(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}
