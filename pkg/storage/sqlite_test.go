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
func TestCheckpointRoundTrip(t *testing.T) {
	ctx := context.Background()
	s, err := NewSQLiteStorage(filepath.Join(t.TempDir(), "nas.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	a := searchspace.DefaultSearchSpace().SampleRandomArchitecture()
	a.Metadata.Fitness = .8
	cp := Checkpoint{Version: 1, Strategy: "random", EvaluationNumber: 1, History: []*searchspace.Architecture{a}, BestFitness: .8, SearchSpaceRNG: 22}
	if err = s.SaveSearchCheckpoint(ctx, "e", cp); err != nil {
		t.Fatal(err)
	}
	got, err := s.LoadLatestCheckpoint(ctx, "e")
	if err != nil {
		t.Fatal(err)
	}
	if got == nil || got.EvaluationNumber != 1 || len(got.History) != 1 {
		t.Fatalf("got=%+v", got)
	}
}

func TestCheckpointRoundTripIncludesExactResumeState(t *testing.T) {
	s, err := NewSQLiteStorage(filepath.Join(t.TempDir(), "state.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	a := searchspace.DefaultSearchSpace().SampleRandomArchitecture()
	a.Metadata.Fitness = 0.5
	cp := Checkpoint{Version: 1, Strategy: "regularized", EvaluationNumber: 1, History: []*searchspace.Architecture{a}, Population: []*searchspace.Architecture{a}, StrategyRNG: 11, SearchSpaceRNG: 22, ConfigJSON: "{}"}
	if err := s.SaveSearchCheckpoint(context.Background(), "exp", cp); err != nil {
		t.Fatal(err)
	}
	got, err := s.LoadLatestCheckpoint(context.Background(), "exp")
	if err != nil {
		t.Fatal(err)
	}
	if got.StrategyRNG != 11 || got.SearchSpaceRNG != 22 || len(got.Population) != 1 || got.ConfigJSON != "{}" {
		t.Fatalf("incomplete checkpoint: %+v", got)
	}
}

func TestCheckpointRejectsLogicalCorruption(t *testing.T) {
	for _, cp := range []Checkpoint{
		{Version: 1, Strategy: "random", EvaluationNumber: 2, History: nil, SearchSpaceRNG: 1},
		{Version: 1, Strategy: "random", EvaluationNumber: 1, History: []*searchspace.Architecture{searchspace.DefaultSearchSpace().SampleRandomArchitecture()}},
		{Version: 1, Strategy: "evolutionary", EvaluationNumber: 1, History: []*searchspace.Architecture{searchspace.DefaultSearchSpace().SampleRandomArchitecture()}, SearchSpaceRNG: 1},
	} {
		s, err := NewSQLiteStorage(filepath.Join(t.TempDir(), "bad.db"))
		if err != nil {
			t.Fatal(err)
		}
		if err := s.SaveSearchCheckpoint(context.Background(), "e", cp); err != nil {
			t.Fatal(err)
		}
		if _, err := s.LoadLatestCheckpoint(context.Background(), "e"); err == nil {
			t.Fatalf("accepted corrupt checkpoint: %+v", cp)
		}
		_ = s.Close()
	}
}
