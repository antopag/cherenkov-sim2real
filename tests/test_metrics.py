"""Tests for ML and physics metrics."""

from __future__ import annotations

import numpy as np

from cherenkov_sim2real.metrics.physics import q_factor


def test_q_factor_perfect() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    # eps_gamma = 1.0, eps_hadron = 0.0 -> Q = 0.0 by convention
    assert q_factor(y_true, y_pred) == 0.0


def test_q_factor_all_gamma() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1])
    # eps_gamma = 1.0, eps_hadron = 1.0 -> Q = 1.0
    assert q_factor(y_true, y_pred) == 1.0


def test_q_factor_partial() -> None:
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    # eps_gamma = 2/4 = 0.5, eps_hadron = 1/4 = 0.25
    # Q = 0.5 / sqrt(0.25) = 0.5 / 0.5 = 1.0
    assert abs(q_factor(y_true, y_pred) - 1.0) < 1e-10


def test_q_factor_no_gammas() -> None:
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 0, 0])
    assert q_factor(y_true, y_pred) == 0.0


def test_q_factor_no_hadrons() -> None:
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 0, 1])
    assert q_factor(y_true, y_pred) == 0.0
