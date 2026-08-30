"""Development-time test rig.

The platform's harness lives in `harness/` and must not be edited: it mirrors the
real protocol and clock. This package copies the parts that need to change for
honest measurement -- opening diversity, parallelism, a stopping rule -- and
leaves the clock and wire protocol byte-identical to `harness/`.
"""
