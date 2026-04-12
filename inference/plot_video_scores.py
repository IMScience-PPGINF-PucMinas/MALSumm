# -*- coding: utf-8 -*-
"""
plot_video_scores.py
====================
Generates one chart per video showing three signals:

  1. Raw model scores   — importance score per frame before knapsack selection,
                          averaged per shot and normalised once (min-max).
  2. Knapsack selection — binary mask (0/1) per shot, derived from the same
                          generate_summary call used to compute F1, so chart
                          and metric are guaranteed to be consistent.
  3. Ground truth       — mean of all annotator summaries, averaged per shot
                          and normalised to [0,1] for visual comparison.

Output
------
One .png file per video saved to <output_dir>/<dataset>/
Filename = sanitised video name (spaces → underscores, special chars removed).

Usage
-----
    # Evaluate a specific checkpoint and plot all test videos
    python -m inference.plot_video_scores \\
        --dataset   SumMe \\
        --split     0 \\
        --epoch     135 \\
        --output_dir plots/

    # Plot every split
    python -m inference.plot_video_scores \\
        --dataset   TVSum \\
        --all_splits \\
        --epoch     best \\          # reads f_scores.txt to find best epoch
        --output_dir plots/
"""

import argparse
import os
import re
import json
import logging

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use('Agg')                       # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

from utils.utils import get_paths
from model.layers.summarizer import xLSTM
from inference.generate_summary import generate_summary
from evaluation.evaluation_metrics import evaluate_summary


# ---------------------------------------------------------------------------
# Palette — consistent across every chart in the report
# ---------------------------------------------------------------------------
PALETTE = {
    'raw':        '#4C72B0',   # muted blue  — model scores
    'knapsack':   '#DD8452',   # warm orange — selected frames
    'gt':         '#55A868',   # green       — ground truth
    'shot_edge':  '#CCCCCC',   # light grey  — shot boundary ticks
    'bg_select':  '#FFF3E0',   # pale amber  — selected region fill
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _min_max_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalise a 1-D array to [0, 1]. Returns zeros if range < eps."""
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < eps:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _sanitise_filename(name: str) -> str:
    """Convert a video name to a safe filename."""
    name = str(name).strip().replace(' ', '_')
    name = re.sub(r'[^\w\-.]', '', name)
    return name or 'video'


# ---------------------------------------------------------------------------
# Single unified pipeline: scores → shot-level signals + F1
# ---------------------------------------------------------------------------

def _build_shot_signals(scores_norm, positions, n_frames,
                        shot_bound, user_summary, eval_metric):
    """Derive all chart signals and F1 from a single generate_summary call.

    This is the **only** place where scores are turned into a binary summary.
    Both the chart and the F1 metric use exactly the same selected frames.

    Parameters
    ----------
    scores_norm  : ndarray [n_picks]  model scores, already min-max normalised
    positions    : ndarray [n_picks]  sub-sampled frame indices
    n_frames     : int
    shot_bound   : ndarray [n_shots, 2]
    user_summary : ndarray [n_annotators, n_frames] or [n_frames]
    eval_metric  : str  'avg' (TVSum) or 'max' (SumMe / MrHiSum)

    Returns
    -------
    shot_model_scores : ndarray [n_shots]  per-shot mean of normalised scores
    knapsack_selected : ndarray [n_shots]  1 = selected, 0 = not (float)
    shot_gt_scores    : ndarray [n_shots]  per-shot GT, normalised to [0,1]
    shot_lengths      : list[int]
    selected_indices  : list[int]          shot indices chosen by knapsack
    f_score           : float              F1 in [0, 100]
    """
    # ------------------------------------------------------------------
    # 1. Upsample sub-sampled scores to full frame resolution
    #    (nearest-neighbour, matching generate_summary.py behaviour)
    # ------------------------------------------------------------------
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    pos = positions.astype(np.int32)
    if pos[-1] != n_frames:
        pos = np.concatenate([pos, [n_frames]])
    for i in range(len(pos) - 1):
        frame_scores[pos[i]:pos[i + 1]] = scores_norm[i] if i < len(scores_norm) else 0.0

    # ------------------------------------------------------------------
    # 2. Per-shot mean model score (what the knapsack actually receives)
    # ------------------------------------------------------------------
    shot_lengths      = [int(s[1] - s[0] + 1) for s in shot_bound]
    shot_model_scores = np.array(
        [frame_scores[s[0]:s[1] + 1].mean() for s in shot_bound],
        dtype=np.float32,
    )

    # ------------------------------------------------------------------
    # 3. Binary summary via generate_summary  ← single source of truth
    #    for both the chart and F1
    # ------------------------------------------------------------------
    summary = generate_summary(
        [shot_bound], [scores_norm.tolist()], [n_frames], [positions]
    )[0]                                       # ndarray [n_frames], dtype bool/int

    # ------------------------------------------------------------------
    # 4. Derive which shots were selected from the binary frame summary
    # ------------------------------------------------------------------
    knapsack_selected = np.zeros(len(shot_bound), dtype=np.float32)
    selected_indices  = []
    for i, s in enumerate(shot_bound):
        if summary[s[0]:s[1] + 1].any():
            knapsack_selected[i] = 1.0
            selected_indices.append(i)

    # ------------------------------------------------------------------
    # 5. F1 score — same summary, same metric as training evaluation
    # ------------------------------------------------------------------
    f_score = evaluate_summary(summary, user_summary, eval_metric)

    # ------------------------------------------------------------------
    # 6. Per-shot GT (mean of annotators, normalised for visual comparison)
    # ------------------------------------------------------------------
    gt_frame       = np.atleast_2d(user_summary).mean(axis=0).astype(np.float32)
    shot_gt_scores = np.array(
        [gt_frame[s[0]:min(s[1] + 1, len(gt_frame))].mean() for s in shot_bound],
        dtype=np.float32,
    )
    shot_gt_scores = _min_max_normalize(shot_gt_scores)

    return (
        shot_model_scores, knapsack_selected,
        shot_gt_scores, shot_lengths,
        selected_indices, f_score,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_video(video_id, shot_model_scores, knapsack_selected,
               shot_gt_scores, shot_lengths, selected_indices,
               shot_bound, video_name, dataset, output_dir, f_score=None):
    """Render and save one chart for a single video at segment (shot) level.

    The X-axis represents shots, not individual frames. Each shot occupies a
    horizontal span proportional to its frame count, giving a faithful temporal
    representation.

    Parameters
    ----------
    video_id          : str
    shot_model_scores : ndarray [n_shots]
    knapsack_selected : ndarray [n_shots]  1 = selected (float)
    shot_gt_scores    : ndarray [n_shots]
    shot_lengths      : list[int]
    selected_indices  : list[int]
    shot_bound        : ndarray [n_shots, 2]
    video_name        : str
    dataset           : str
    output_dir        : str
    f_score           : float | None
    """
    n_shots      = len(shot_model_scores)
    total_frames = sum(shot_lengths)

    shot_starts  = np.array([shot_bound[i][0]     for i in range(n_shots)], dtype=float)
    shot_ends    = np.array([shot_bound[i][1] + 1  for i in range(n_shots)], dtype=float)
    shot_centers = (shot_starts + shot_ends) / 2.0

    # ---- figure layout ----
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=130)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#F5F5F5')

    # ---- background: selected-shot spans ----
    for idx in selected_indices:
        ax.axvspan(shot_starts[idx], shot_ends[idx],
                   facecolor=PALETTE['bg_select'],
                   alpha=0.50, zorder=1, linewidth=0)

    # ---- shot boundary lines ----
    for i in range(n_shots - 1):
        ax.axvline(shot_ends[i], color=PALETTE['shot_edge'],
                   linewidth=0.5, zorder=2, alpha=0.7)

    # ---- GT per shot ----
    ax.step(shot_starts, shot_gt_scores,
            color=PALETTE['gt'], linewidth=1.4,
            where='post', alpha=0.85, zorder=3,
            label='Ground truth (mean, per shot)')

    # ---- model scores per shot ----
    ax.step(shot_starts, shot_model_scores,
            color=PALETTE['raw'], linewidth=1.4,
            where='post', alpha=0.90, zorder=4,
            label='Model score (normalised, per shot)')

    # ---- knapsack binary selection ----
    ax.step(shot_starts, knapsack_selected,
            color=PALETTE['knapsack'], linewidth=2.0,
            where='post', alpha=0.95, zorder=5,
            label='Knapsack selection (1 = selected)')

    # ---- markers at centres of selected shots ----
    if selected_indices:
        sel_x = shot_centers[selected_indices]
        sel_y = shot_model_scores[selected_indices]
        ax.scatter(sel_x, sel_y,
                   color=PALETTE['knapsack'], s=28,
                   zorder=6, alpha=0.85)

    # ---- axes formatting ----
    ax.set_xlim(0, total_frames)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Frame position (shot boundaries)', fontsize=10, labelpad=6)
    ax.set_ylabel('Score / Selection', fontsize=10, labelpad=6)

    step            = max(1, n_shots // 8)
    tick_positions  = shot_starts[::step]
    tick_labels     = [str(i * step) for i in range(len(tick_positions))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(shot_starts))
    ax.tick_params(axis='x', which='minor', length=2, color=PALETTE['shot_edge'])

    ax.text(0.5, -0.13,
            f'{n_shots} shots  ·  {total_frames:,} frames total',
            transform=ax.transAxes, ha='center',
            fontsize=8, color='#777777')

    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=8.5, length=3)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    # ---- legend ----
    legend_handles = [
        plt.Line2D([0], [0], color=PALETTE['raw'],
                   linewidth=1.5, label='Model score (normalised, per shot)'),
        plt.Line2D([0], [0], color=PALETTE['knapsack'],
                   linewidth=2.0, label='Knapsack selection (1 = selected)'),
        plt.Line2D([0], [0], color=PALETTE['gt'],
                   linewidth=1.5, label='Ground truth (mean, per shot)'),
        Patch(facecolor=PALETTE['bg_select'], edgecolor='none',
              alpha=0.7, label='Selected region'),
    ]
    ax.legend(handles=legend_handles,
              loc='upper right', fontsize=8,
              framealpha=0.85, edgecolor='#CCCCCC',
              handlelength=2.0)

    # ---- title ----
    title_parts = [f'{dataset}  ·  {video_name}  ({video_id})']
    if f_score is not None:
        title_parts.append(f'F1 = {f_score:.2f}%')
    ax.set_title('   |   '.join(title_parts),
                 fontsize=10.5, fontweight='bold',
                 pad=8, loc='left')

    # ---- annotation: selection stats ----
    pct_shots  = len(selected_indices) / n_shots * 100
    pct_frames = sum(shot_lengths[i] for i in selected_indices) / total_frames * 100
    ax.text(0.995, 1.02,
            f'Selected: {len(selected_indices)}/{n_shots} shots '
            f'({pct_shots:.0f}%)  ·  {pct_frames:.1f}% of frames',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=8, color='#555555')

    # ---- save ----
    out_folder = os.path.join(output_dir, dataset)
    os.makedirs(out_folder, exist_ok=True)
    safe_name  = _sanitise_filename(video_name if video_name else video_id)
    out_path   = os.path.join(out_folder, f'{safe_name}.png')

    plt.tight_layout(pad=1.4)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Per-split driver
# ---------------------------------------------------------------------------

def plot_split(split_id, dataset, model_path, epoch_fname,
               dataset_path, split_data, model_kwargs,
               output_dir, verbose=False):
    """Load one checkpoint and generate charts for all its test videos.

    Parameters
    ----------
    split_id     : int
    dataset      : str
    model_path   : str
    epoch_fname  : str   e.g. 'epoch-135.pkl'
    dataset_path : str
    split_data   : list | dict
    model_kwargs : dict
    output_dir   : str
    verbose      : bool

    Returns
    -------
    list[str]: paths of saved chart files
    """
    eval_metric = 'avg' if dataset.lower() == 'tvsum' else 'max'
    test_keys   = (
        split_data[split_id]['test_keys']
        if isinstance(split_data, list)
        else split_data['test_keys']
    )

    ckpt_path = os.path.join(model_path, epoch_fname)
    model = xLSTM(**model_kwargs)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()

    saved_paths = []

    with h5py.File(dataset_path, 'r') as hdf:
        for video_id in test_keys:
            # Skip out-of-range SumMe videos
            if dataset.lower() == 'summe':
                try:
                    if int(video_id.split('_')[1]) > 25:
                        continue
                except (IndexError, ValueError):
                    pass

            # --- Load h5 fields ---
            features     = torch.Tensor(np.array(hdf[f'{video_id}/features'])).view(-1, 1024)
            shot_bound   = np.array(hdf[f'{video_id}/change_points'])
            n_frames     = int(np.array(hdf[f'{video_id}/n_frames']))
            positions    = np.array(hdf[f'{video_id}/picks'])

            if f'{video_id}/user_summary' in hdf:
                user_summary = np.array(hdf[f'{video_id}/user_summary'])
            elif f'{video_id}/gt_summary' in hdf:
                user_summary = np.array(hdf[f'{video_id}/gt_summary'])
            else:
                logging.warning(f'No ground truth found for {video_id} — skipping')
                continue

            video_name = video_id
            if f'{video_id}/video_name' in hdf:
                video_name = str(
                    np.array(hdf[f'{video_id}/video_name']).astype(str, copy=False)
                )

            # --- Model inference ---
            with torch.no_grad():
                scores, _ = model(features)
                scores = scores.squeeze(0).cpu().numpy()

            # --- Single normalisation pass (never repeated downstream) ---
            scores_norm = _min_max_normalize(scores)

            # --- Unified pipeline: signals + F1 from the same binary summary ---
            (shot_model_scores, knapsack_selected,
             shot_gt_scores, shot_lengths,
             selected_indices, f_score) = _build_shot_signals(
                scores_norm, positions, n_frames,
                shot_bound, user_summary, eval_metric,
            )

            # --- Plot ---
            out_path = plot_video(
                video_id          = video_id,
                shot_model_scores = shot_model_scores,
                knapsack_selected = knapsack_selected,
                shot_gt_scores    = shot_gt_scores,
                shot_lengths      = shot_lengths,
                selected_indices  = selected_indices,
                shot_bound        = shot_bound,
                video_name        = video_name,
                dataset           = dataset,
                output_dir        = output_dir,
                f_score           = f_score,
            )
            saved_paths.append(out_path)

            if verbose:
                logging.info(f'  {video_id} ({video_name}) → {out_path}  F1={f_score:.2f}%')
            else:
                print(f'  Saved: {out_path}  F1={f_score:.2f}%')

    return saved_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _resolve_epoch_fname(model_path, epoch_arg):
    """Return the checkpoint filename for a given epoch argument.

    epoch_arg can be:
      - 'best'   → reads f_scores.txt and returns the best epoch file
      - int str  → returns 'epoch-{N}.pkl'
    """
    if str(epoch_arg).lower() == 'best':
        fscores_path = os.path.join(model_path, 'f_scores.txt')
        if not os.path.exists(fscores_path):
            raise FileNotFoundError(
                f"f_scores.txt not found in {model_path}. "
                "Run compute_fscores.py first, or pass --epoch <N>."
            )
        with open(fscores_path) as fp:
            content = fp.read().strip()
        try:
            scores = json.loads(content)
        except json.JSONDecodeError:
            scores = [float(x) for x in content.splitlines()]
        best = int(np.argmax(scores))
        return f'epoch-{best}.pkl', best

    epoch_num = int(epoch_arg)
    return f'epoch-{epoch_num}.pkl', epoch_num


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-video score charts for a trained xLSTM checkpoint.'
    )
    parser.add_argument('--dataset',       type=str,  default='SumMe',
                        help='Dataset [SumMe | TVSum | MrHiSum]')
    parser.add_argument('--model_version', type=str,  default='',
                        help='Model version suffix')
    parser.add_argument('--split',         type=int,  default=0,
                        help='Split index to plot (ignored if --all_splits)')
    parser.add_argument('--all_splits',    action='store_true',
                        help='Plot all 5 splits')
    parser.add_argument('--epoch',         type=str,  default='best',
                        help="Epoch to load: integer or 'best' (reads f_scores.txt)")
    parser.add_argument('--output_dir',    type=str,  default='plots',
                        help='Root directory for output charts')
    parser.add_argument('--verbose',       action='store_true')
    parser.add_argument('--hidden_dim',    type=int,  default=512)
    parser.add_argument('--num_layers',    type=int,  default=2)
    parser.add_argument('--dropout',       type=float, default=0.5)

    args = vars(parser.parse_args())

    dataset       = args['dataset']
    model_version = args['model_version']
    output_dir    = args['output_dir']
    verbose       = args['verbose']
    split_ids     = (
        list(range(5))
        if args['all_splits']
        else [args['split']]
    )

    model_kwargs = dict(
        input_size=1024,
        output_size=1024,
        num_segments=4,
        hidden_dim=args['hidden_dim'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
    )

    paths        = get_paths(dataset)
    dataset_path = paths['dataset']
    split_file   = paths['split']

    with open(split_file) as fp:
        split_data = json.load(fp)

    total_saved = []

    for split_id in split_ids:
        model_path = (
            f"Summaries/xLSTM/{dataset}{model_version}/models/split{split_id}"
        )
        try:
            epoch_fname, epoch_num = _resolve_epoch_fname(model_path, args['epoch'])
        except FileNotFoundError as e:
            logging.error(str(e))
            continue

        print(
            f"\nSplit {split_id} — epoch {epoch_num} — "
            f"generating charts for {dataset}..."
        )

        saved = plot_split(
            split_id     = split_id,
            dataset      = dataset,
            model_path   = model_path,
            epoch_fname  = epoch_fname,
            dataset_path = dataset_path,
            split_data   = split_data,
            model_kwargs = model_kwargs,
            output_dir   = output_dir,
            verbose      = verbose,
        )
        total_saved.extend(saved)

    print(f"\nDone. {len(total_saved)} chart(s) saved to '{output_dir}/'")


if __name__ == '__main__':
    main()
