package main

import (
	"nas-go/pkg/utils"
	"testing"
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
