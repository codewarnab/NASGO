package utils

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfigMergesDefaultsAndYAML(t *testing.T) {
	p := filepath.Join(t.TempDir(), "config.yml")
	if err := os.WriteFile(p, []byte("search:\n  max_evaluations: 7\nlogging:\n  level: debug\n"), 0644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if c.Search.MaxEvaluations != 7 || c.Search.Strategy == "" || c.Logging.Level != "debug" {
		t.Fatalf("unexpected config: %+v", c)
	}
}
func TestConfigValidation(t *testing.T) {
	c := DefaultConfig()
	c.Search.Strategy = "bogus"
	if c.Validate() == nil {
		t.Fatal("expected invalid strategy")
	}
}
func TestEnvironmentOverridesYAML(t *testing.T) {
	p := filepath.Join(t.TempDir(), "config.yml")
	if err := os.WriteFile(p, []byte("search:\n  max_evaluations: 7\n"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("NAS_MAX_EVALUATIONS", "9")
	t.Setenv("NAS_SEARCH_STRATEGY", "random")
	t.Setenv("NAS_USE_GPU", "false")
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatal(err)
	}
	if c.Search.MaxEvaluations != 9 || c.Search.Strategy != "random" || c.Evaluator.UseGPU {
		t.Fatalf("unexpected: %+v", c)
	}
}
func TestInvalidEnvironmentFails(t *testing.T) {
	p := filepath.Join(t.TempDir(), "config.yml")
	if err := os.WriteFile(p, []byte("{}"), 0644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("NAS_NUM_WORKERS", "many")
	if _, err := LoadConfig(p); err == nil {
		t.Fatal("expected parse error")
	}
}
