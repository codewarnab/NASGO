package storage

import (
	"context"
	"math"
	"path/filepath"
	"testing"
	"time"

	"nas-go/pkg/searchspace"
)

func TestSQLiteExperimentArchitectureAndCheckpoint(t *testing.T) {
	ctx := context.Background()
	s, err := NewSQLiteStorage(filepath.Join(t.TempDir(), "nas.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	exp := Experiment{ID: "e1", Name: "fixture", ConfigJSON: "{}", Strategy: "random", StartedAt: time.Now()}
	if err = s.CreateExperiment(ctx, exp); err != nil {
		t.Fatal(err)
	}
	got, err := s.GetExperiment(ctx, "e1")
	if err != nil || got == nil || got.Name != "fixture" {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	a := searchspace.DefaultSearchSpace().SampleRandomArchitecture()
	a.Metadata.Fitness = .7
	if err = s.SaveArchitecture(ctx, "e1", a); err != nil {
		t.Fatal(err)
	}
	b := searchspace.DefaultSearchSpace().SampleRandomArchitecture()
	b.Metadata.Fitness = .9
	if err = s.SaveArchitecture(ctx, "e1", b); err != nil {
		t.Fatal(err)
	}
	top, err := s.GetTopArchitectures(ctx, "e1", 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(top) != 2 || math.Abs(top[0].Metadata.Fitness-.9) > .0001 {
		t.Fatalf("unexpected order: %+v", top)
	}
	if err = s.SaveCheckpoint(ctx, "e1", 2, top, .9); err != nil {
		t.Fatal(err)
	}
}
