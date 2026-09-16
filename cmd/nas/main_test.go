package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"nas-go/pkg/storage"
	"nas-go/pkg/utils"
)

func TestBuildSearchSpaceAndSearcher(t *testing.T) {
	c := utils.DefaultConfig()
	s, err := buildSearchSpace(c)
	if err != nil || s.Size() <= 0 {
		t.Fatalf("space err=%v", err)
	}
	for _, name := range []string{"random", "evolutionary", "regularized"} {
		c.Search.Strategy = name
		if _, err := buildSearcher(c); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
}

func captureStdout(t *testing.T, fn func() error) (string, error) {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	done := make(chan string, 1)
	go func() {
		var b bytes.Buffer
		_, _ = io.Copy(&b, r)
		done <- b.String()
	}()
	runErr := fn()
	_ = w.Close()
	os.Stdout = old
	out := <-done
	_ = r.Close()
	return out, runErr
}

func writeCLIConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunSearchConfigAndFlagPrecedence(t *testing.T) {
	path := writeCLIConfig(t, `
search:
  strategy: random
  max_evaluations: 2
  population_size: 2
  tournament_size: 1
  num_workers: 1
  search_space:
    num_nodes: 2
    num_input_nodes: 2
    edges_per_node: 1
    operations: [identity, zero]
evaluator:
  type: proxy
storage:
  type: none
  save_history: false
logging:
  level: error
  format: text
`)
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{name: "file values", args: []string{"--config", path}, want: "Evaluations: 2 | Strategy: random"},
		{name: "flags override file", args: []string{"--config", path, "--strategy", "evolutionary", "--evaluations", "3", "--population", "2"}, want: "Evaluations: 3 | Strategy: evolutionary"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, err := captureStdout(t, func() error { return runSearch(tc.args) })
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(out, tc.want) {
				t.Fatalf("output does not contain %q:\n%s", tc.want, out)
			}
		})
	}
}

func TestRunSearchDefaultsWithFlagOverrides(t *testing.T) {
	db := filepath.Join(t.TempDir(), "search.db")
	out, err := captureStdout(t, func() error {
		return runSearch([]string{"--strategy", "random", "--evaluations", "1", "--evaluator", "proxy", "--db", db, "--log-level", "error"})
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "Evaluations: 1 | Strategy: random") {
		t.Fatalf("flags did not override defaults:\n%s", out)
	}
	if _, err := os.Stat(db); err != nil {
		t.Fatalf("expected search database: %v", err)
	}
}

func TestRunSearchFailures(t *testing.T) {
	badOperation := writeCLIConfig(t, `
search:
  strategy: random
  max_evaluations: 1
  population_size: 1
  search_space:
    operations: [not_an_operation]
storage:
  type: none
`)
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{name: "unknown flag", args: []string{"--not-a-flag"}, want: "flag provided but not defined"},
		{name: "missing config", args: []string{"--config", filepath.Join(t.TempDir(), "missing.yaml")}, want: "loading config"},
		{name: "invalid strategy", args: []string{"--strategy", "bogus", "--evaluations", "1"}, want: "invalid configuration"},
		{name: "bad search space", args: []string{"--config", badOperation}, want: "building search space"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := captureStdout(t, func() error { return runSearch(tc.args) })
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error=%v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestRunInfoConfigAndFailures(t *testing.T) {
	valid := writeCLIConfig(t, `
experiment:
  seed: 7
search:
  search_space:
    num_nodes: 2
    num_input_nodes: 2
    edges_per_node: 1
    operations: [identity, zero]
`)
	out, err := captureStdout(t, func() error { return runInfo([]string{"--config", valid}) })
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "Nodes per cell:     2") || !strings.Contains(out, "Operations:         identity, zero") {
		t.Fatalf("unexpected info output:\n%s", out)
	}

	bad := writeCLIConfig(t, "search:\n  search_space:\n    operations: [bogus]\n")
	for _, args := range [][]string{{"--config", bad}, {"--config", filepath.Join(t.TempDir(), "missing.yaml")}, {"--unknown"}} {
		if _, err := captureStdout(t, func() error { return runInfo(args) }); err == nil {
			t.Fatalf("runInfo(%v) unexpectedly succeeded", args)
		}
	}
}

func TestRunSearchEnvironmentAndCLIPrecedenceWithoutConfigFlag(t *testing.T) {
	t.Setenv("NAS_CONFIG", "")
	t.Setenv("NAS_MAX_EVALUATIONS", "2")
	t.Setenv("NAS_SEARCH_STRATEGY", "random")
	t.Setenv("NAS_STORAGE_PATH", filepath.Join(t.TempDir(), "env.db"))
	out, err := captureStdout(t, func() error {
		return runSearch([]string{"--evaluations", "1", "--log-level", "error"})
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "Evaluations: 1 | Strategy: random") {
		t.Fatalf("expected CLI > environment > defaults:\n%s", out)
	}
}

func TestRunInfoUsesNASConfigFallback(t *testing.T) {
	path := writeCLIConfig(t, `
search:
  search_space:
    num_nodes: 2
    num_input_nodes: 2
    edges_per_node: 1
    operations: [identity, zero]
`)
	t.Setenv("NAS_CONFIG", path)
	out, err := captureStdout(t, func() error { return runInfo(nil) })
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "Nodes per cell:     2") {
		t.Fatalf("NAS_CONFIG fallback not used:\n%s", out)
	}
}

func writeResumeConfig(t *testing.T, db string, maxEvaluations, checkpointInterval int) string {
	t.Helper()
	return writeCLIConfig(t, fmt.Sprintf(`
experiment:
  name: resume-test
  seed: 42
search:
  strategy: random
  max_evaluations: %d
  population_size: 2
  tournament_size: 1
  num_workers: 1
  search_space:
    num_nodes: 2
    num_input_nodes: 2
    edges_per_node: 1
    operations: [identity, zero]
evaluator:
  type: proxy
storage:
  type: sqlite
  path: %q
  save_history: true
  checkpoint_interval: %d
logging:
  level: error
  format: text
`, maxEvaluations, db, checkpointInterval))
}

func TestRunSearchPeriodicCheckpointAndResume(t *testing.T) {
	db := filepath.Join(t.TempDir(), "resume.db")
	configPath := writeResumeConfig(t, db, 2, 1)
	if _, err := captureStdout(t, func() error { return runSearch([]string{"--config", configPath}) }); err != nil {
		t.Fatal(err)
	}

	database, err := sql.Open("sqlite", db)
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	var experimentID string
	if err := database.QueryRow(`SELECT id FROM experiments LIMIT 1`).Scan(&experimentID); err != nil {
		t.Fatal(err)
	}
	var count int
	if err := database.QueryRow(`SELECT count(*) FROM checkpoints WHERE experiment_id=?`, experimentID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count == 0 {
		t.Fatal("periodic checkpoint was not saved")
	}

	resumePath := writeResumeConfig(t, db, 3, 1)
	if _, err := captureStdout(t, func() error { return runSearch([]string{"--config", resumePath, "--resume", experimentID}) }); err == nil || !strings.Contains(err.Error(), "incompatible") {
		t.Fatalf("changed budget should fail compatibility check, got %v", err)
	}
	if _, err := captureStdout(t, func() error { return runSearch([]string{"--config", configPath, "--resume", experimentID}) }); err != nil {
		t.Fatalf("same-config resume failed: %v", err)
	}
	if err := database.QueryRow(`SELECT count(*) FROM experiments WHERE id=?`, experimentID).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("resume duplicated experiment row: %d", count)
	}
}

func TestRunSearchResumeRejectsMissingAndCorruptCheckpoints(t *testing.T) {
	db := filepath.Join(t.TempDir(), "invalid.db")
	configPath := writeResumeConfig(t, db, 2, 1)
	store, err := storage.NewSQLiteStorage(db)
	if err != nil {
		t.Fatal(err)
	}
	cfg, err := utils.LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	configJSON, _ := cfg.ToJSON()
	for _, id := range []string{"missing-checkpoint", "corrupt-checkpoint"} {
		if err := store.CreateExperiment(context.Background(), storage.Experiment{ID: id, Name: id, ConfigJSON: string(configJSON), Strategy: "random", StartedAt: time.Now()}); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.SaveSearchCheckpoint(context.Background(), "corrupt-checkpoint", storage.Checkpoint{Version: 1, Strategy: "random", EvaluationNumber: 2, History: nil, SearchSpaceRNG: 1, ConfigJSON: string(configJSON)}); err != nil {
		t.Fatal(err)
	}
	_ = store.Close()
	for _, tc := range []struct{ id, want string }{{"absent", "not found"}, {"missing-checkpoint", "no checkpoint"}, {"corrupt-checkpoint", "inconsistent checkpoint"}} {
		_, err := captureStdout(t, func() error { return runSearch([]string{"--config", configPath, "--resume", tc.id}) })
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("resume %s: err=%v want %q", tc.id, err, tc.want)
		}
	}
}

func TestRunSearchCancellationSavesCheckpoint(t *testing.T) {
	db := filepath.Join(t.TempDir(), "cancel.db")
	dir := t.TempDir()
	script := filepath.Join(dir, "slow.py")
	marker := filepath.Join(dir, "marker")
	scriptBody := fmt.Sprintf("import os,time\np=%q\nif not os.path.exists(p):\n open(p,'w').close(); print('{\"validation_accuracy\":0.5}')\nelse:\n time.sleep(10)\n", marker)
	if err := os.WriteFile(script, []byte(scriptBody), 0o600); err != nil {
		t.Fatal(err)
	}
	configPath := writeCLIConfig(t, fmt.Sprintf(`
experiment: {name: cancel-test, seed: 42}
search:
  strategy: random
  max_evaluations: 10
  population_size: 2
  tournament_size: 1
  num_workers: 1
  search_space: {num_nodes: 2, num_input_nodes: 2, edges_per_node: 1, operations: [identity, zero]}
evaluator:
  type: trainer
  dataset: fake
  script_path: %q
  python_path: python3
  timeout: 1m
storage:
  type: sqlite
  path: %q
  save_history: true
  checkpoint_interval: 1
logging: {level: error, format: text}
`, script, db))
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()
	_, err := captureStdout(t, func() error { return runSearchContext(ctx, []string{"--config", configPath}) })
	if err != nil {
		t.Fatalf("graceful cancellation returned error: %v", err)
	}
	database, err := sql.Open("sqlite", db)
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	var checkpoints int
	if err := database.QueryRow(`SELECT count(*) FROM checkpoints`).Scan(&checkpoints); err != nil {
		t.Fatal(err)
	}
	// The timed-out trainer result is a completed evaluation; cancellation must preserve a coherent final checkpoint.
	var status string
	if err := database.QueryRow(`SELECT status FROM experiments LIMIT 1`).Scan(&status); err != nil {
		t.Fatal(err)
	}
	if status != "cancelled" || checkpoints == 0 {
		t.Fatalf("status=%s checkpoints=%d", status, checkpoints)
	}
}

func TestCheckpointConfigJSONIsValid(t *testing.T) {
	cfg := utils.DefaultConfig()
	data, err := cfg.ToJSON()
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatal(err)
	}
	if len(out) == 0 {
		t.Fatal("empty config JSON")
	}
}
