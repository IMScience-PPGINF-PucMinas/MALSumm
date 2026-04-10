# -*- coding: utf-8 -*-
"""
plot_video_scores.py
====================
Generates one chart per video showing three signals:

  1. Raw model scores   — importance score per frame before knapsack selection,
                          upsampled to the original frame resolution.
  2. Knapsack selection — binary mask (0/1) produced by the knapsack algorithm,
                          shown as a filled step plot so selected regions are
                          immediately visible.
  3. Ground truth       — mean of all annotator summaries, smoothed slightly
                          so the reference signal is readable alongside the others.

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
from scipy.ndimage import uniform_filter1d  # lightweight smoothing for GT

from utils.utils import get_paths
from model.layers.summarizer import xLSTM
from inference.generate_summary import generate_summary
from inference.knapsack_implementation import knapSack


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
# Score pipeline helpers
# ---------------------------------------------------------------------------

def _upsample_scores(scores, positions, n_frames):
    """Expand sub-sampled scores to full-resolution frame scores.

    Mirrors the logic in generate_summary.py so the raw score line aligns
    exactly with the knapsack and GT signals.
    """
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    pos = positions.astype(np.int32)
    if pos[-1] != n_frames:
        pos = np.concatenate([pos, [n_frames]])
    for i in range(len(pos) - 1):
        val = scores[i] if i < len(scores) else 0.0
        frame_scores[pos[i]:pos[i + 1]] = val
    return frame_scores


def _compute_knapsack_scores(frame_scores, shot_bound):
    """Reproduce the shot-level importance + knapsack selection.

    Returns
    -------
    knapsack_mask : np.ndarray [n_frames] binary, 1 = selected
    shot_scores   : list of float, one per shot (for optional annotation)
    """
    shot_imp_scores, shot_lengths = [], []
    for shot in shot_bound:
        shot_lengths.append(int(shot[1] - shot[0] + 1))
        shot_imp_scores.append(float(frame_scores[shot[0]:shot[1] + 1].mean()))

    final_max_length = int((shot_bound[-1][1] + 1) * 0.15)
    selected = knapSack(
        final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths)
    )

    mask = np.zeros(shot_bound[-1][1] + 1, dtype=np.int8)
    for idx in selected:
        mask[shot_bound[idx][0]:shot_bound[idx][1] + 1] = 1

    return mask, shot_imp_scores


def _mean_gt(user_summary):
    """Return per-frame mean importance across all annotators.

    user_summary shape: [n_annotators, n_frames] or [n_frames].
    Smoothed with a small uniform filter so the reference curve is
    readable even when individual annotations are very sparse/binary.
    """
    gt = np.atleast_2d(user_summary).mean(axis=0).astype(np.float32)
    # Smooth over ~2% of video length, minimum 5 frames
    window = max(5, int(len(gt) * 0.02))
    return uniform_filter1d(gt, size=window)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _sanitise_filename(name):
    """Convert a video name to a safe filename."""
    name = str(name).strip()
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\-.]', '', name)
    return name or 'video'


def plot_video(video_id, raw_scores, knapsack_mask, gt_mean,
               shot_bound, video_name, dataset, output_dir, f_score=None):
    """Render and save one chart for a single video.

    Parameters
    ----------
    video_id      : str   h5 key, e.g. 'video_1'
    raw_scores    : ndarray [n_frames]  model scores 0-1
    knapsack_mask : ndarray [n_frames]  binary selection
    gt_mean       : ndarray [n_frames]  smoothed GT mean
    shot_bound    : ndarray [n_shots,2] shot boundaries
    video_name    : str   human-readable name (used as title + filename)
    dataset       : str   'SumMe' | 'TVSum' | …
    output_dir    : str   root output directory
    f_score       : float optional — shown in title if provided
    """
    n_frames = len(raw_scores)
    x        = np.arange(n_frames)

    # ---- figure layout ----
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=130)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#F5F5F5')

    # ---- shot boundary vertical lines ----
    for shot in shot_bound:
        for edge in (shot[0], shot[1]):
            ax.axvline(edge, color=PALETTE['shot_edge'],
                       linewidth=0.4, zorder=1, alpha=0.6)

    # ---- knapsack selected regions (filled background) ----
    in_region  = False
    region_start = 0
    extended = np.append(knapsack_mask, 0)      # sentinel to close last region
    for i, val in enumerate(extended):
        if val == 1 and not in_region:
            region_start = i
            in_region    = True
        elif val == 0 and in_region:
            ax.axvspan(region_start, i,
                       facecolor=PALETTE['bg_select'],
                       alpha=0.45, zorder=2, linewidth=0)
            in_region = False

    # ---- ground truth (mean annotators, smoothed) ----
    gt_plot = gt_mean[:n_frames]
    ax.plot(x[:len(gt_plot)], gt_plot,
            color=PALETTE['gt'], linewidth=1.4,
            alpha=0.85, zorder=3, label='Ground truth (mean annotators)')

    # ---- raw model scores ----
    ax.plot(x, raw_scores,
            color=PALETTE['raw'], linewidth=1.2,
            alpha=0.9, zorder=4, label='Model scores (before knapsack)')

    # ---- knapsack binary mask (step) ----
    ax.step(x, knapsack_mask.astype(np.float32),
            color=PALETTE['knapsack'], linewidth=1.5,
            where='post', zorder=5, alpha=0.9,
            label='Knapsack selection (binary)')

    # ---- axes formatting ----
    ax.set_xlim(0, n_frames - 1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Frame index', fontsize=10, labelpad=6)
    ax.set_ylabel('Score / Selection', fontsize=10, labelpad=6)

    # X-axis: show ~8 evenly spaced ticks
    x_ticks = np.linspace(0, n_frames - 1, num=8, dtype=int)
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda val, _: f'{int(val):,}'
    ))

    # Y-axis: 0.0, 0.25, 0.50, 0.75, 1.0
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    ax.tick_params(axis='both', labelsize=8.5, length=3)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)

    # ---- legend ----
    legend_handles = [
        plt.Line2D([0], [0], color=PALETTE['raw'],
                   linewidth=1.5, label='Model scores (before knapsack)'),
        plt.Line2D([0], [0], color=PALETTE['knapsack'],
                   linewidth=1.5, label='Knapsack selection (binary)'),
        plt.Line2D([0], [0], color=PALETTE['gt'],
                   linewidth=1.5, label='Ground truth (mean annotators)'),
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

    # ---- annotation: % frames selected ----
    pct = knapsack_mask.mean() * 100
    ax.text(0.995, 1.02,
            f'Selected: {pct:.1f}% of frames',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=8, color='#555555')

    # ---- save ----
    out_folder = os.path.join(output_dir, dataset)
    os.makedirs(out_folder, exist_ok=True)
    safe_name  = _sanitise_filename(video_name if video_name else video_id)
    out_path   = os.path.join(out_folder, f'{safe_name}.png')

    plt.tight_layout(pad=1.2)
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
    model_path   : str   path to folder containing epoch .pkl files
    epoch_fname  : str   filename of the checkpoint to load, e.g. 'epoch-135.pkl'
    dataset_path : str   path to the h5 dataset file
    split_data   : list|dict  loaded split json
    model_kwargs : dict  constructor args for xLSTM
    output_dir   : str
    verbose      : bool

    Returns
    -------
    list of str: paths of saved chart files
    """
    eval_metric = 'avg' if dataset.lower() == 'tvsum' else 'max'
    test_keys   = (
        split_data[split_id]['test_keys']
        if isinstance(split_data, list)
        else split_data['test_keys']
    )

    # Load model
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
                scores_tensor, _ = model(features)
                scores = scores_tensor.squeeze(0).cpu().numpy().tolist()

            # --- Build three signals ---
            raw_scores    = _upsample_scores(scores, positions, n_frames)
            knapsack_mask, _ = _compute_knapsack_scores(raw_scores, shot_bound)
            gt_mean       = _mean_gt(user_summary)

            # --- Optional F1 for title ---
            from evaluation.evaluation_metrics import evaluate_summary
            from inference.generate_summary import generate_summary as _gs
            summary  = _gs([shot_bound], [scores], [n_frames], [positions])[0]
            if(dataset == 'TVSum'):
                f_score  = evaluate_summary(summary, user_summary, 'avg')
            else:
                f_score  = evaluate_summary(summary, user_summary, 'max')

            # --- Plot ---
            out_path = plot_video(
                video_id      = video_id,
                raw_scores    = raw_scores,
                knapsack_mask = knapsack_mask,
                gt_mean       = gt_mean,
                shot_bound    = shot_bound,
                video_name    = video_name,
                dataset       = dataset,
                output_dir    = output_dir,
                f_score       = f_score,
            )
            saved_paths.append(out_path)

            if verbose:
                logging.info(f'  {video_id} ({video_name}) → {out_path}  F1={f_score:.2f}%')
            else:
                print(f'  Saved: {out_path}')

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
        if args['all_splits'] or dataset.lower() in ('summe', 'tvsum') and args['all_splits']
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
