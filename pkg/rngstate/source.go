// Package rngstate supplies a small deterministic rand.Source whose complete
// state can be persisted in search checkpoints.
package rngstate

type Source struct{ State uint64 }

func New(seed int64) *Source { s := &Source{}; s.Seed(seed); return s }
func (s *Source) Seed(seed int64) {
	s.State = uint64(seed)
	if s.State == 0 {
		s.State = 0x9e3779b97f4a7c15
	}
}
func (s *Source) Uint64() uint64 {
	x := s.State
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	s.State = x
	return x * 2685821657736338717
}
func (s *Source) Int63() int64 { return int64(s.Uint64() >> 1) }
