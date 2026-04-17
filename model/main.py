# -*- coding: utf-8 -*-
"""main_cv.py — Cross-validation training entry point.

Replaces the single split_index loop in main.py with a full k-fold CV loop.
For each fold:
  1. Build train/test loaders using the CV-aware data loader.
  2. Instantiate a fresh Solver (fresh model + optimizer).
  3. Train and evaluate exactly as before.

The fold index is injected into config so that Solver saves checkpoints in
  Summaries/xLSTM/<dataset>/models/fold{fold_id}/
matching the paths expected by inference_cv.py.
"""

import json
import numpy as np
from sklearn.model_selection import KFold

from configs.configs import get_config
from data.data_loader_cv import get_loader_cv   # CV-aware loader (see data_loader_cv.py)
from solver_cv import SolverCV                  # CV-aware solver   (see solver_cv.py)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_all_keys(split_file: str) -> list[str]:
    """Return the sorted union of every video key in the original split JSON."""
    with open(split_file) as fp:
        split_data = json.load(fp)

    keys = set()
    if isinstance(split_data, list):
        for entry in split_data:
            for k in ("train_keys", "test_keys", "val_keys"):
                keys.update(entry.get(k, []))
    else:
        for k in ("train_keys", "test_keys", "val_keys"):
            keys.update(split_data.get(k, []))

    return sorted(keys)


def build_cv_folds(all_keys: list[str], n_folds: int = 5, seed: int = 42):
    """Partition *all_keys* into n_folds deterministic (train, test) pairs."""
    keys_arr = np.array(all_keys)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in kf.split(keys_arr):
        folds.append({
            "train_keys": keys_arr[train_idx].tolist(),
            "test_keys":  keys_arr[test_idx].tolist(),
        })
    return folds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Base configs (mode / hyperparams come from here as usual)
    # ------------------------------------------------------------------
    config      = get_config(mode='train')
    test_config = get_config(mode='test')

    # CV parameters — add these to your config or set them here directly
    n_folds = getattr(config, 'n_folds', 5)
    cv_seed = getattr(config, 'cv_seed', 42)

    print(config)
    print(f"\nCross-validation: {n_folds} folds  |  seed: {cv_seed}\n")

    # ------------------------------------------------------------------
    # Build fold definitions from the original split file
    # ------------------------------------------------------------------
    all_keys = collect_all_keys(config.split_file)   # config must expose split_file path
    cv_folds = build_cv_folds(all_keys, n_folds=n_folds, seed=cv_seed)

    print(f"Total videos: {len(all_keys)}  "
          f"|  ~{len(cv_folds[0]['test_keys'])} test videos per fold\n")

    # ------------------------------------------------------------------
    # Train one model per fold
    # ------------------------------------------------------------------
    fold_results = []

    for fold_id, fold in enumerate(cv_folds):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold_id}  —  "
              f"train: {len(fold['train_keys'])} videos  "
              f"test: {len(fold['test_keys'])} videos")
        print(f"{'='*60}\n")

        # Inject fold_id so Solver saves to  models/fold{fold_id}/
        config.fold_id      = fold_id
        test_config.fold_id = fold_id

        # Build data loaders for this fold using explicit key lists
        train_loader = get_loader_cv(
            mode=config.mode,
            video_type=config.video_type,
            keys=fold["train_keys"],
        )
        test_loader = get_loader_cv(
            mode=test_config.mode,
            video_type=test_config.video_type,
            keys=fold["test_keys"],
        )

        # Fresh model + optimizer for every fold
        solver = SolverCV(config, test_config, train_loader, test_loader, fold_id)
        solver.build()
        solver.evaluate(-1)   # baseline with random weights
        solver.train()

        fold_results.append({
            "fold_id": fold_id,
            "best_fscore": solver.best_fscore,   # SolverCV tracks this
        })

    # ------------------------------------------------------------------
    # Final cross-validation summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold_id']}:  best F1 = {r['best_fscore']:.2f}%")
    scores = [r["best_fscore"] for r in fold_results]
    print(f"  {'─'*40}")
    print(f"  Mean F1: {np.mean(scores):.2f}%  ±  {np.std(scores):.2f}%")
    print(f"{'='*60}\n")
