package evaluator

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"nas-go/pkg/searchspace"
)

func testArch() *searchspace.Architecture {
	return searchspace.DefaultSearchSpace().SampleRandomArchitecture()
}

type fixedEvaluator struct {
	name    string
	fitness float64
	err     error
}

func (f fixedEvaluator) Evaluate(context.Context, *searchspace.Architecture) (*EvaluationResult, error) {
	return &EvaluationResult{Fitness: f.fitness}, f.err
}
func (f fixedEvaluator) Name() string                 { return f.name }
func (f fixedEvaluator) EstimatedTime() time.Duration { return time.Millisecond }

func TestProxyEvaluatorProducesFiniteFitness(t *testing.T) {
	r, err := NewProxyEvaluator(DefaultProxyConfig()).Evaluate(context.Background(), testArch())
	if err != nil {
		t.Fatal(err)
	}
	if r.Fitness < 0 || r.Fitness > 1 {
		t.Fatalf("fitness=%v", r.Fitness)
	}
}
func TestCombinedEvaluatorStopsAtThreshold(t *testing.T) {
	c := NewCombinedEvaluator([]Evaluator{fixedEvaluator{"first", .1, nil}, fixedEvaluator{"second", .9, nil}}, []float64{.5})
	r, err := c.Evaluate(context.Background(), testArch())
	if err != nil {
		t.Fatal(err)
	}
	if r.Fitness != .1 || !strings.Contains(r.Error, "below threshold") {
		t.Fatalf("unexpected result: %+v", r)
	}
}
func TestTrainerResultParsing(t *testing.T) {
	tr := &TrainerEvaluator{}
	r, err := tr.parseResults([]byte("log\n{\"accuracy\":0.8,\"validation_accuracy\":0.75,\"epochs\":2}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if r.Fitness != .75 || r.Accuracy != .8 || r.Epochs != 2 {
		t.Fatalf("unexpected: %+v", r)
	}
	if _, err = tr.parseResults([]byte("not json")); err == nil {
		t.Fatal("expected malformed-output error")
	}
}
func TestTrainerSubprocessProtocol(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fixture uses a shell script")
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "trainer.py")
	body := "import sys\nprint( '{\"validation_accuracy\":0.91,\"accuracy\":0.92,\"epochs\":1}')\n"
	if err := os.WriteFile(script, []byte(body), 0755); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultTrainerConfig()
	cfg.Timeout = time.Second
	cfg.TempDir = dir
	tr, err := NewTrainerEvaluator(cfg, script, "python3")
	if err != nil {
		t.Fatal(err)
	}
	r, err := tr.Evaluate(context.Background(), testArch())
	if err != nil {
		t.Fatal(err)
	}
	if r.Fitness != .91 {
		t.Fatalf("fitness=%v", r.Fitness)
	}
}
func TestTrainerSubprocessFailure(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fixture uses a shell script")
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "trainer.py")
	if err := os.WriteFile(script, []byte("import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n"), 0644); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultTrainerConfig()
	cfg.TempDir = dir
	tr, err := NewTrainerEvaluator(cfg, script, "python3")
	if err != nil {
		t.Fatal(err)
	}
	_, err = tr.Evaluate(context.Background(), testArch())
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("unexpected error: %v", err)
	}
}
