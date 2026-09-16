package search

import (
	"context"
	"sync"
	"time"

	"nas-go/pkg/searchspace"
)

type batchOutcome struct {
	arch     *searchspace.Architecture
	fitness  float64
	duration time.Duration
	err      error
}

// evaluateBatch evaluates at most NumWorkers candidates concurrently. It stops
// dispatching when the context is cancelled and always joins in-flight work.
func evaluateBatch(ctx context.Context, workers int, arches []*searchspace.Architecture, evaluate func(context.Context, *searchspace.Architecture) (float64, error)) []batchOutcome {
	if workers < 1 {
		workers = 1
	}
	if workers > len(arches) {
		workers = len(arches)
	}
	outcomes := make([]batchOutcome, len(arches))
	for i, arch := range arches {
		outcomes[i] = batchOutcome{arch: arch, err: context.Canceled}
	}
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				if ctx.Err() != nil {
					outcomes[i] = batchOutcome{arch: arches[i], err: ctx.Err()}
					continue
				}
				began := time.Now()
				fitness, err := evaluate(ctx, arches[i])
				outcomes[i] = batchOutcome{arch: arches[i], fitness: fitness, duration: time.Since(began), err: err}
			}
		}()
	}
	for i := range arches {
		if ctx.Err() != nil {
			break
		}
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return outcomes
}
