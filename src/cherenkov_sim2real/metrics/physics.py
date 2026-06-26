"""Physics-driven metrics for gamma-hadron separation.

See PLAN.md S6.2 and the glossary in S9.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def q_factor(
    y_true: npt.NDArray[np.int_],
    y_pred: npt.NDArray[np.int_],
) -> float:
    """Compute the Q-factor for gamma-hadron separation.

    Q = epsilon_gamma / sqrt(epsilon_hadron)

    where epsilon_gamma is the gamma efficiency (fraction of true gammas
    predicted as gamma) and epsilon_hadron is the hadron acceptance
    (fraction of true hadrons predicted as gamma).

    Parameters
    ----------
    y_true : array of int
        Ground-truth labels (1 = gamma, 0 = hadron).
    y_pred : array of int
        Predicted labels (1 = gamma, 0 = hadron).

    Returns
    -------
    float
        Q-factor. Higher is better. Returns 0.0 if epsilon_hadron is 0
        (no hadrons predicted as gamma) or if there are no gammas/hadrons.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    gamma_mask = y_true == 1
    hadron_mask = y_true == 0

    n_gamma = gamma_mask.sum()
    n_hadron = hadron_mask.sum()

    if n_gamma == 0 or n_hadron == 0:
        return 0.0

    eps_gamma = float(y_pred[gamma_mask].sum()) / float(n_gamma)
    eps_hadron = float(y_pred[hadron_mask].sum()) / float(n_hadron)

    if eps_hadron <= 0.0:
        return 0.0

    return float(eps_gamma / np.sqrt(eps_hadron))
