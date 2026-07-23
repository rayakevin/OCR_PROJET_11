"""Tests unitaires des fonctions pures de la campagne Optuna.

Ces fonctions décident de la forme de la recherche (échantillonnage de gamma,
grille de raffinement, déduplication des finalistes, cloisonnement des seeds).
Une régression y fausserait un résultat sans lever d'erreur : elles sont donc
testées en isolation, sans lancer le moindre entraînement.
"""

import math

import numpy as np
import pytest
from optuna.trial import FixedTrial

import optuna_search as search
import optuna_select as select
import optuna_final as final


# --- gamma_from_trial : reparamétrisation log(1 - gamma) --------------------

@pytest.mark.parametrize("one_minus_gamma", [1e-4, 1e-3, 5e-3, 2e-2])
def test_gamma_from_trial_inverts_one_minus_gamma(one_minus_gamma):
    """gamma vaut exactement 1 - one_minus_gamma, sans erreur de signe."""
    trial = FixedTrial({"one_minus_gamma": one_minus_gamma})
    assert search.gamma_from_trial(trial) == pytest.approx(1.0 - one_minus_gamma)


def test_gamma_from_trial_spans_expected_range():
    """Les bornes de l'espace donnent gamma entre 0.98 et 0.9999."""
    low = search.gamma_from_trial(FixedTrial({"one_minus_gamma": 2e-2}))
    high = search.gamma_from_trial(FixedTrial({"one_minus_gamma": 1e-4}))
    assert low == pytest.approx(0.98)
    assert high == pytest.approx(0.9999)
    assert low < high  # un one_minus_gamma plus grand donne un gamma plus petit


# --- resolved_parameters : traduction essai -> arguments PPO -----------------

def test_resolved_parameters_replaces_one_minus_gamma_with_gamma():
    resolved = search.resolved_parameters(
        {"one_minus_gamma": 2.4e-4, "learning_rate": 1e-3, "n_steps": 2048, "batch_size": 128}
    )
    assert "one_minus_gamma" not in resolved
    assert resolved["gamma"] == pytest.approx(1.0 - 2.4e-4)
    assert resolved["learning_rate"] == 1e-3


def test_resolved_parameters_clamps_batch_size_to_n_steps():
    """PPO exige batch_size <= n_steps : un dépassement est ramené à n_steps."""
    resolved = search.resolved_parameters(
        {"one_minus_gamma": 1e-3, "n_steps": 128, "batch_size": 256}
    )
    assert resolved["batch_size"] == 128


def test_resolved_parameters_leaves_valid_batch_size_untouched():
    resolved = search.resolved_parameters(
        {"one_minus_gamma": 1e-3, "n_steps": 2048, "batch_size": 128}
    )
    assert resolved["batch_size"] == 128


# --- SearchConfig : budget et notation multi-seed ---------------------------

def test_total_timesteps_multiplies_by_seed_count():
    config = search.SearchConfig(timesteps_per_seed=100_000, train_seeds=(1, 2, 3))
    assert config.total_timesteps == 300_000


def test_default_search_scores_each_trial_on_several_seeds():
    """Le cœur de la refonte : un essai est jugé sur plusieurs seeds, jamais une.

    Une valeur d'essai obtenue sur une seule seed mesurerait le bruit
    d'initialisation autant que le réglage.
    """
    assert len(search.SearchConfig().train_seeds) >= 2


# --- Cloisonnement des seeds ------------------------------------------------

def test_search_selection_and_test_seed_ranges_are_disjoint():
    """Recherche, sélection et évaluation finale tirent sur des plages séparées.

    Sélectionner ou annoncer un résultat sur des seeds vues pendant la recherche
    donnerait une performance optimiste.
    """
    search_eval = search.SEARCH_EVAL_SEED_START            # 2000+
    selection = select.SELECTION_EVAL_SEED_START           # 5000+
    test = final.FINAL_SEED_START                          # 10000+
    starts = [search_eval, selection, test]
    assert len(set(starts)) == 3
    assert search_eval < selection < test
    # Marge suffisante pour que les épisodes d'une phase n'atteignent pas la suivante.
    assert selection - search_eval >= 1000
    assert test - selection >= 1000


def test_final_selection_never_uses_test_seeds():
    """La sélection du modèle final se fait hors du jeu de test réservé."""
    assert final.SELECTION_SEED_START != final.FINAL_SEED_START
    assert final.SELECTION_SEED_START < final.FINAL_SEED_START


def test_final_training_seeds_differ_from_robustness_seeds():
    """L'entraînement final rejoue des seeds neuves, pas celles de la robustesse."""
    assert set(final.FINAL_TRAIN_SEEDS).isdisjoint(select.ROBUST_TRAIN_SEEDS)


# --- build_gamma_grid : grille centrée, régulière en horizon effectif -------

def test_gamma_grid_is_centered_on_measured_optimum():
    """La valeur médiane de la grille retombe sur l'optimum fourni."""
    best = 0.99976
    grid = select.build_gamma_grid(best, 9)
    assert grid[len(grid) // 2] == pytest.approx(best, abs=1e-6)


def test_gamma_grid_is_regular_in_log_not_in_gamma():
    """La grille est régulière en log(1 - gamma), franchement pas en gamma.

    C'est ce qui donne autant de résolution des deux côtés de l'optimum ; une
    grille régulière en gamma écraserait le côté des horizons courts. On compare
    la dispersion relative des pas dans les deux échelles plutôt que d'exiger
    l'égalité exacte, car les valeurs sont arrondies à six décimales.
    """
    grid = np.array(select.build_gamma_grid(0.9997, 9))

    def coefficient_of_variation(values):
        steps = np.abs(np.diff(values))
        return float(np.std(steps) / np.mean(steps))

    log_steps_cv = coefficient_of_variation(np.log10(1.0 - grid))
    gamma_steps_cv = coefficient_of_variation(grid)

    assert log_steps_cv < 0.02          # pas réguliers en log(1 - gamma)
    assert gamma_steps_cv > 0.5         # très irréguliers en gamma
    assert log_steps_cv < gamma_steps_cv


def test_gamma_grid_spans_one_decade_each_side():
    """La grille explore un facteur 10 de (1 - gamma) de part et d'autre."""
    best = 0.9997
    grid = select.build_gamma_grid(best, 9)
    center = 1.0 - best
    assert (1.0 - grid[0]) == pytest.approx(center * 10, rel=1e-6)
    assert (1.0 - grid[-1]) == pytest.approx(center / 10, rel=1e-6)


def test_gamma_grid_stays_within_valid_bounds():
    """Aucune valeur hors de l'intervalle ouvert utile."""
    grid = select.build_gamma_grid(0.9999, 9)
    assert all(0.9 < g < 0.99999 for g in grid)


def test_gamma_grid_values_are_sorted_and_unique():
    grid = select.build_gamma_grid(0.99976, 9)
    assert grid == sorted(grid)
    assert len(grid) == len(set(grid))


# --- dedup_key : l'incident des deux finalistes identiques ------------------

def test_dedup_key_merges_the_real_incident_pair():
    """Rejoue l'incident : deux gammas à 5e-7 près donnent la même clé.

    `broad_t068` (gamma en pleine précision) et `gamma_g004` (même gamma arrondi
    à six décimales par la grille) désignaient le même réglage mais étaient
    validés deux fois. La déduplication doit les fondre en un seul finaliste.
    """
    base = {"learning_rate": 0.001691, "n_steps": 2048, "batch_size": 128}
    broad = select.dedup_key({**base, "gamma": 0.9997595239264913})
    grid = select.dedup_key({**base, "gamma": 0.99976})
    assert broad == grid


def test_dedup_key_keeps_distinct_grid_points_separate():
    """Deux points de grille voisins restent deux finalistes distincts."""
    base = {"learning_rate": 0.001691, "n_steps": 2048, "batch_size": 128}
    grid = select.build_gamma_grid(0.99976, 9)
    assert select.dedup_key({**base, "gamma": grid[3]}) != select.dedup_key(
        {**base, "gamma": grid[4]}
    )


def test_dedup_key_absorbs_float_noise_on_other_parameters():
    """Un écart flottant négligeable sur un autre paramètre ne scinde pas la clé."""
    a = select.dedup_key({"gamma": 0.99976, "learning_rate": 0.001691})
    b = select.dedup_key({"gamma": 0.99976, "learning_rate": 0.001691 + 1e-12})
    assert a == b


def test_dedup_key_distinguishes_different_learning_rates():
    a = select.dedup_key({"gamma": 0.99976, "learning_rate": 1e-3})
    b = select.dedup_key({"gamma": 0.99976, "learning_rate": 2e-3})
    assert a != b
