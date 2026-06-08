import numpy as np

from active_learning.acquisition import select_top_uncertain


def test_selects_highest_uncertainty_first():
    names = ["a", "b", "c", "d"]
    unc = np.array([0.1, 0.9, 0.5, 0.2])
    picked = select_top_uncertain(names, unc, k=2, exclude=set())
    assert picked == ["b", "c"]


def test_excludes_attempted_and_respects_k():
    names = ["a", "b", "c", "d"]
    unc = np.array([0.1, 0.9, 0.5, 0.2])
    picked = select_top_uncertain(names, unc, k=2, exclude={"b"})
    assert picked == ["c", "d"]


def test_k_larger_than_available():
    names = ["a", "b"]
    unc = np.array([0.4, 0.7])
    picked = select_top_uncertain(names, unc, k=10, exclude=set())
    assert picked == ["b", "a"]
