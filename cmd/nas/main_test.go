package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
