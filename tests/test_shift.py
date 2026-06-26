"""Tests for domain-shift estimators.

Cover at least:
- MMD == 0 (within tolerance) for two samples drawn from the same
  distribution, and grows monotonically with mean shift,
- proxy-A-distance ≈ 0 for indistinguishable domains and ≈ 2 for
  trivially separable ones,
- ECE drift sign matches an obvious miscalibration.
"""
