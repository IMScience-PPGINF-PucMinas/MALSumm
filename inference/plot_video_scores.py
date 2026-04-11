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
# Score pipeline helpers — segment (shot) level
# ---------------------------------------------------------------------------

def _compute_shot_signals(scores, positions, n_frames, shot_bound, user_summary):
    """Compute all three signals at shot (segment) level.

    O eixo X do gráfico representa segmentos (shots), não frames individuais.
    Isso alinha diretamente com a unidade de decisão do knapsack e torna
    visível quais segmentos foram selecionados versus seus scores originais.

    Parameters
    ----------
    scores       : list[float]  scores subamostrados do modelo (len = n_picks)
    positions    : ndarray      índices de frames subamostrados (picks)
    n_frames     : int          total de frames do vídeo original
    shot_bound   : ndarray      [n_shots, 2] boundaries de cada shot
    user_summary : ndarray      [n_annotators, n_frames] ou [n_frames]

    Returns
    -------
    shot_model_scores : ndarray [n_shots]  score médio do modelo por shot
                        (normalizado min-max — mesmo valor que entra no knapsack)
    knapsack_selected : ndarray [n_shots]  1 = selecionado pelo knapsack, 0 = não
    shot_gt_scores    : ndarray [n_shots]  importância média do GT por shot
    shot_lengths      : list[int]          comprimento em frames de cada shot
    selected_indices  : list[int]          índices dos shots selecionados
    """
    # 1. Upsample scores subamostrados → todos os frames
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    pos = positions.astype(np.int32)
    if pos[-1] != n_frames:
        pos = np.concatenate([pos, [n_frames]])
    for i in range(len(pos) - 1):
        frame_scores[pos[i]:pos[i + 1]] = scores[i] if i < len(scores) else 0.0

    # 2. Normalização min-max (igual ao generate_summary.py atualizado)
    s_min, s_max = frame_scores.min(), frame_scores.max()
    denom = s_max - s_min
    if denom > 1e-8:
        frame_scores = (frame_scores - s_min) / denom

    # 3. Score médio do modelo por shot (valor que o knapsack recebe)
    shot_model_scores = np.array([
        float(frame_scores[s[0]:s[1] + 1].mean())
        for s in shot_bound
    ], dtype=np.float32)

    shot_lengths = [int(s[1] - s[0] + 1) for s in shot_bound]

    # 4. Knapsack: seleciona shots que maximizam score dentro de 15% do vídeo
    final_max_length = int((shot_bound[-1][1] + 1) * 0.15)
    selected_indices = knapSack(
        final_max_length, shot_lengths,
        shot_model_scores.tolist(), len(shot_lengths)
    )

    knapsack_selected = np.zeros(len(shot_bound), dtype=np.float32)
    for idx in selected_indices:
        knapsack_selected[idx] = 1.0

    # 5. GT médio por shot — média dos anotadores, depois média por shot
    gt_frame = np.atleast_2d(user_summary).mean(axis=0).astype(np.float32)
    shot_gt_scores = np.array([
        float(gt_frame[s[0]:min(s[1] + 1, len(gt_frame))].mean())
        for s in shot_bound
    ], dtype=np.float32)

    # Normaliza GT para [0,1] para facilitar comparação visual
    gt_min, gt_max = shot_gt_scores.min(), shot_gt_scores.max()
    if gt_max - gt_min > 1e-8:
        shot_gt_scores = (shot_gt_scores - gt_min) / (gt_max - gt_min)

    return (shot_model_scores, knapsack_selected,
            shot_gt_scores, shot_lengths, selected_indices)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _sanitise_filename(name):
    """Convert a video name to a safe filename."""
    name = str(name).strip()
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\-.]', '', name)
    return name or 'video'


def plot_video(video_id, shot_model_scores, knapsack_selected,
               shot_gt_scores, shot_lengths, selected_indices,
               shot_bound, video_name, dataset, output_dir, f_score=None):
    """Render and save one chart for a single video at segment (shot) level.

    O eixo X representa segmentos (shots), não frames individuais.
    Cada ponto no eixo X corresponde a um shot, e sua largura visual é
    proporcional ao comprimento do shot em frames — dando uma representação
    temporal fiel mesmo no espaço de segmentos.

    Parameters
    ----------
    video_id          : str      h5 key, e.g. 'video_1'
    shot_model_scores : ndarray  [n_shots] score médio normalizado por shot
    knapsack_selected : ndarray  [n_shots] 1=selecionado, 0=não (float)
    shot_gt_scores    : ndarray  [n_shots] GT médio normalizado por shot
    shot_lengths      : list     comprimento em frames de cada shot
    selected_indices  : list     índices dos shots selecionados
    shot_bound        : ndarray  [n_shots, 2] para referência
    video_name        : str
    dataset           : str
    output_dir        : str
    f_score           : float opcional
    """
    n_shots      = len(shot_model_scores)
    total_frames = sum(shot_lengths)

    # Posição central de cada shot em frames (eixo X proporcional ao tempo)
    # Cada shot ocupa uma faixa proporcional ao seu comprimento
    shot_starts  = np.array([shot_bound[i][0] for i in range(n_shots)], dtype=float)
    shot_ends    = np.array([shot_bound[i][1] + 1 for i in range(n_shots)], dtype=float)
    shot_centers = (shot_starts + shot_ends) / 2.0

    # ---- figure layout ----
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=130)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#F5F5F5')

    # ---- fundo: regiões selecionadas pelo knapsack ----
    # Destacadas como faixas verticais proporcionais ao comprimento do shot
    for idx in selected_indices:
        ax.axvspan(shot_starts[idx], shot_ends[idx],
                   facecolor=PALETTE['bg_select'],
                   alpha=0.50, zorder=1, linewidth=0)

    # ---- linhas de fronteira entre shots ----
    for i in range(n_shots - 1):
        ax.axvline(shot_ends[i], color=PALETTE['shot_edge'],
                   linewidth=0.5, zorder=2, alpha=0.7)

    # ---- GT médio por shot (step centrado na posição do shot) ----
    ax.step(shot_starts, shot_gt_scores,
            color=PALETTE['gt'], linewidth=1.4,
            where='post', alpha=0.85, zorder=3,
            label='Ground truth (mean, per shot)')

    # ---- Score do modelo por shot (antes da mochila) ----
    ax.step(shot_starts, shot_model_scores,
            color=PALETTE['raw'], linewidth=1.4,
            where='post', alpha=0.90, zorder=4,
            label='Model score (normalised, per shot)')

    # ---- Seleção binária do knapsack ----
    ax.step(shot_starts, knapsack_selected,
            color=PALETTE['knapsack'], linewidth=2.0,
            where='post', alpha=0.95, zorder=5,
            label='Knapsack selection (1 = selected)')

    # ---- marcadores nos centros dos shots selecionados ----
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

    # X-axis: ticks nos inícios de shots espaçados regularmente (~8 ticks)
    step = max(1, n_shots // 8)
    tick_positions = shot_starts[::step]
    tick_labels    = [str(i * step) for i in range(len(tick_positions))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(shot_starts))
    ax.tick_params(axis='x', which='minor', length=2,
                   color=PALETTE['shot_edge'], alpha=0.6)

    # Segunda linha no eixo X com número total de shots
    ax.text(0.5, -0.13,
            f'{n_shots} shots  ·  {total_frames:,} frames total',
            transform=ax.transAxes, ha='center',
            fontsize=8, color='#777777')

    # Y-axis
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

    # ---- annotation: n shots + % selecionados ----
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

def _min_max_normalize(scores, eps=1e-8):
    scores = np.array(scores, dtype=float)
    min_val = scores.min()
    max_val = scores.max()

    if max_val - min_val < eps:
        return np.zeros_like(scores)

    return (scores - min_val) / (max_val - min_val)

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
                scores, _, _, _ = model(features)
                scores = scores.squeeze(0).cpu().numpy()

            # ----------------------------------
            # NORMALIZAÇÃO (igual à inferência)
            # ----------------------------------
            scores = _min_max_normalize(scores)

            # manter compatibilidade
            scores = scores.tolist()

            # --- Build three signals at shot level ---
            (shot_model_scores, knapsack_selected,
             shot_gt_scores, shot_lengths,
             selected_indices) = _compute_shot_signals(
                scores, positions, n_frames, shot_bound, user_summary
            )

            # --- Optional F1 for title ---
            from evaluation.evaluation_metrics import evaluate_summary
            from inference.generate_summary import generate_summary as _gs
            summary  = _gs([shot_bound], [scores], [n_frames], [positions])[0]
            if dataset == 'TVSum':
                f_score = evaluate_summary(summary, user_summary, 'avg')
            else:
                f_score = evaluate_summary(summary, user_summary, 'max')

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
