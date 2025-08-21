import torch
from os import listdir
import numpy as np
from os.path import join
import h5py
import json
import argparse
import cv2
import os
import logging
import pandas as pd
import re
from utils.utils import get_paths, setup_logging
from evaluation.evaluation_metrics import evaluate_summary
from model.layers.summarizer import xLSTM
from .generate_summary import generate_summary
from scipy.stats import kendalltau, spearmanr
from openpyxl import load_workbook
from openpyxl.styles import Alignment

setup_logging()

def load_video_data(dataset, data_path, video):
    """Load video data from the dataset."""
    with h5py.File(data_path, "r") as hdf:
        frame_features = torch.Tensor(
            np.array(hdf[f"{video}/features"])
        ).view(-1, 1024)
        sb = np.array(hdf[f"{video}/change_points"])
        video_name = None

        if dataset.lower() in ('summe', 'tvsum'):
            user_summary = np.array(hdf[f"{video}/user_summary"])
            n_frames     = np.array(hdf[f"{video}/n_frames"])
            positions    = np.array(hdf[f"{video}/picks"])

            if "video_name" in hdf[f"{video}"]:
                video_name = str(
                    np.array(hdf[f"{video}/video_name"]).astype(str, copy=False)
                )
        elif dataset.lower() == 'mrhisum':
            user_summary = np.array(hdf[f"{video}/gt_summary"])
            n_frames     = frame_features.shape[0]
            positions    = np.arange(n_frames, dtype=int)
            
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    return frame_features, user_summary, sb, n_frames, positions, video_name


def save_video_frames(video_summaries, video_names, split_id, dataset, save_frames=True):
    """Save summarized video frames and videos."""
    for video, summary_indices in video_summaries.items():
        video_name = video if dataset == 'TVSum' else video_names[video].replace(" ", "_")
        frames_folder = os.path.join(f"data/summarized_frames/{dataset}/{video}")
        os.makedirs(frames_folder, exist_ok=True)

        first_frame_path = os.path.join(f"data/frames/{dataset}/{video_name}", 'img_00001.jpg')
        first_frame = cv2.imread(first_frame_path)
        if first_frame is None:
            logging.error(f"First frame of video {video_name} not found.")
            continue

        frame_height, frame_width, _ = first_frame.shape
        total_frame_quantity = len(summary_indices)
        generated_frame_quantity = 0

        video_path = f'{video}_{video_name}_summary.mp4'
        if os.path.exists(video_path):
            print(f"Video {video_path} already exists. Skipping...")
            continue

        out = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            60.0,
            (frame_width, frame_height)
        )
        print(f"Processing video: {video_name} - SPLIT {split_id}")

        for index, is_selected in enumerate(summary_indices):
            if is_selected == 1:
                frame_path = os.path.join(
                    f"data/frames/{video_name}",
                    f'img_{index+1:05d}.jpg'
                )
                frame = cv2.imread(frame_path)
                if frame is not None:
                    generated_frame_quantity += 1
                    out.write(frame)
                    if save_frames:
                        frame_save_path = os.path.join(
                            frames_folder,
                            f'img_{index+1:05d}.jpg'
                        )
                        cv2.imwrite(frame_save_path, frame)

        print(f"Original frame quantity for {video_name}: {total_frame_quantity}")
        print(f"Summarized frame quantity for {video_name}: {generated_frame_quantity}")
        print(f"Generated frames percentage: {(generated_frame_quantity / total_frame_quantity) * 100:.2f}%")
        out.release()
        print(f"Saved summarized video frames in {frames_folder}")


# def run_inference(model, data_path, keys, eval_method, save_summary, verbose=False):
def run_inference(model, data_path, keys, eval_method, save_summary, dataset, verbose=False):
    """
    Run inference on the dataset, computing F-score, Kendall's tau, and Spearman's rho per video.
    Returns:
      - mean_fscore: average F-score over all videos in this split
      - mean_kendall: average Kendall's tau over all videos in this split
      - mean_spearman: average Spearman's rho over all videos in this split
      - video_summaries: dict mapping video→binary summary array
      - (optionally) video_names if on SumMe
    """
    model.eval()

    video_fscores    = []
    video_kendalls   = []
    video_spearmans  = []
    video_summaries  = {}
    video_names      = {}
    summe = (dataset.lower() == 'summe')

    for video in keys:
        video_number = int(video.split('_')[1])
        if summe and video_number > 25:
            print(f"Skipping video {video}...")
            continue

        frame_features, user_summary, sb, n_frames, positions, video_name = load_video_data(dataset, data_path, video)

        with torch.no_grad():
            scores, _ = model(frame_features)
            scores = scores.squeeze(0).cpu().numpy().tolist()

            summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
            f_score = evaluate_summary(summary, user_summary, eval_method)

            # upsample
            frame_init_scores = np.array(scores)
            frame_scores = np.zeros(n_frames, dtype=float)

            pos = positions.astype(int)
            if pos[-1] != n_frames:
                pos = np.concatenate([pos, [n_frames]])

            for i in range(len(pos) - 1):
                frame_scores[pos[i]:pos[i+1]] = frame_init_scores[i]

            if user_summary.ndim > 1:
                gt_importance = user_summary.mean(axis=0)
            else:
                gt_importance = user_summary

            if frame_scores.shape[0] != gt_importance.shape[0]:
                logging.warning(
                    f"Skipping correlations for {video}: "
                    f"pred len={frame_scores.shape[0]}, gt len={gt_importance.shape[0]}"
                )
                ktau, spr = float('nan'), float('nan')
            else:
                ktau, _ = kendalltau(frame_scores, gt_importance)
                spr,  _ = spearmanr(frame_scores, gt_importance)

            video_fscores.append(f_score)
            video_kendalls.append(ktau)
            video_spearmans.append(spr)
            video_summaries[video] = summary

            if verbose:
                logging.info(f"Summary for video {video} ({video_name}): {summary}")
                logging.info(f"F-score for video {video_name}: {f_score:.2f}%")
                logging.info(f"Kendall τ for video {video_name}: {ktau:.4f}")
                logging.info(f"Spearman ρ for video {video_name}: {spr:.4f}")

            if summe:
                video_names[video] = video_name

            if save_summary:
                summary_json = {str(i): int(frame) for i, frame in enumerate(summary)}
                json_filename = f"{video}_summary.json"
                with open(json_filename, "w") as json_file:
                    json.dump(summary_json, json_file, indent=4)
                print(f"Summary exported to {json_filename}")

    mean_fscore    = float(np.nanmean(video_fscores))
    mean_kendall   = float(np.nanmean(video_kendalls))
    mean_spearman  = float(np.nanmean(video_spearmans))

    if summe:
        return mean_fscore, mean_kendall, mean_spearman, video_summaries, video_names
    else:
        return mean_fscore, mean_kendall, mean_spearman, video_summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='SumMe', help="Dataset to be used. Supported: {SumMe, TVSum}")
    parser.add_argument("--model_version", type=str, default='', help="Model version. Used when multiple versions are available. Example: 'attV5-20e'")
    parser.add_argument("--test_dataset", type=str, default='SumMe', help="Dataset to be used for testing. Supported: {SumMe, TVSum}")
    parser.add_argument("--table", type=str, default='4', help="Table to be reproduced. Supported: {3, 4}")
    parser.add_argument("--best_fscore_only", type=bool, default=True, help="Generate summarized video for the best F-score only.")
    parser.add_argument("--sum_split", type=int, default=None, help="Which split to summarize.")
    parser.add_argument("--verbose", type=int, default=False, help="Debugging logs.")
    parser.add_argument("--save_video", type=int, default=False, help="Whether to save the videos.")
    parser.add_argument("--save_summary", type=int, default=False, help="Whether to save the summary.")

    args = vars(parser.parse_args())
    dataset          = args["dataset"]
    model_version    = args["model_version"]
    test_dataset     = args["test_dataset"]
    best_fscore_only = args["best_fscore_only"]
    sum_split        = args["sum_split"]
    verbose          = args["verbose"]
    save_video       = args["save_video"]
    save_summary     = args["save_summary"]

    eval_metric = 'avg' if dataset.lower() == 'tvsum' else 'max'

    print(f"Running inference for {dataset} dataset, testing with {test_dataset} dataset")

    if dataset.lower() in ('summe','tvsum'):
        split_ids = list(range(5))
    else:
        split_ids = [0]

    epoch_fscores   = {s: {} for s in split_ids}
    epoch_kendalls  = {s: {} for s in split_ids}
    epoch_spearmans = {s: {} for s in split_ids}

    for split_id in split_ids:
        model_path = f"Summaries/xLSTM/{dataset}{model_version}/models/split{split_id}"
        all_epoch_files = sorted(
            [f for f in listdir(model_path) if re.match(r"epoch-\d+\.pkl", f)],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )

        paths = get_paths(dataset)
        split_file = paths['split']
        with open(split_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            test_keys = data[split_id]["test_keys"]
        else:
            test_keys = data["test_keys"]

        dataset_path = paths['dataset']

        for file in all_epoch_files:
            epoch_num = int(re.findall(r'\d+', file)[0])
            model = xLSTM(
                input_size=1024,
                output_size=1024,
                num_segments=4,
                hidden_dim=512,
                num_layers=2,
                dropout=0.2
            )
            model.load_state_dict(torch.load(join(model_path, file)))

            if dataset == 'SumMe':
                fscore, kendall, spearman, _, _ = run_inference(
                    model, dataset_path, test_keys, eval_metric, save_summary, dataset, verbose
                )
            else:
                fscore, kendall, spearman, _ = run_inference(
                    model, dataset_path, test_keys, eval_metric, save_summary, dataset, verbose
                )

            epoch_fscores[split_id][epoch_num]   = fscore
            epoch_kendalls[split_id][epoch_num]  = kendall
            epoch_spearmans[split_id][epoch_num] = spearman

    all_epochs = sorted({e for d in epoch_fscores.values() for e in d})
    data = {"Epoch": all_epochs}
    avg_fs, avg_ks, avg_ss = {}, {}, {}
    for ep in all_epochs:
        fs = [epoch_fscores[s].get(ep) for s in split_ids]
        ks = [epoch_kendalls[s].get(ep) for s in split_ids]
        ss = [epoch_spearmans[s].get(ep) for s in split_ids]
        avg_fs[ep] = np.nanmean([x for x in fs if x is not None])
        avg_ks[ep] = np.nanmean([x for x in ks if x is not None])
        avg_ss[ep] = np.nanmean([x for x in ss if x is not None])
    for split in split_ids:
        data[f"F-score Split {split}"]  = [epoch_fscores[split].get(ep) for ep in all_epochs]
        data[f"Kendall Split {split}"]  = [epoch_kendalls[split].get(ep) for ep in all_epochs]
        data[f"Spearman Split {split}"] = [epoch_spearmans[split].get(ep) for ep in all_epochs]
    data["Avg F-score"]  = [avg_fs[ep] for ep in all_epochs]
    data["Avg Kendall"]  = [avg_ks[ep] for ep in all_epochs]
    data["Avg Spearman"] = [avg_ss[ep] for ep in all_epochs]

    df = pd.DataFrame(data)
    df.index = df["Epoch"]
    df.drop(columns=["Epoch"], inplace=True)

    tuples = []
    for split in split_ids:
        for metric in ("F-score", "Kendall", "Spearman"):
            tuples.append((f"Split {split}", metric))
    for metric in ("F-score", "Kendall", "Spearman"):
        tuples.append(("Average", metric))
    df.columns = pd.MultiIndex.from_tuples(tuples)

    excel_path = f"{dataset}_epoch_metrics.xlsx"
    df.to_excel(excel_path)
    print(f"Saved epoch metrics to {excel_path}")

    wb = load_workbook(excel_path)
    ws = wb.active

    ws.merge_cells("A1:A2")
    hcell = ws["A1"]
    hcell.value = "Epoch"
    hcell.alignment = Alignment(horizontal="center", vertical="center")

    # remove the old "Epoch" row that pandas inserted as header row 3
    ws.delete_rows(3, 1)
    wb.save(excel_path)

    best_epoch = max(avg_fs, key=avg_fs.get)
    best_fscore = avg_fs[best_epoch]
    best_kendall = avg_ks.get(best_epoch, float('nan'))
    best_spearman = avg_ss.get(best_epoch, float('nan'))

    for split in split_ids:
        f_val = epoch_fscores[split].get(best_epoch, None)
        k_val = epoch_kendalls[split].get(best_epoch, None)
        s_val = epoch_spearmans[split].get(best_epoch, None)
        print(f"Split {split} - Epoch {best_epoch}: F={f_val:.2f}%, τ={k_val:.4f}, ρ={s_val:.4f}")

    print(
        f"Best Epoch (avg over splits): Epoch {best_epoch} | "
        f"Avg F-score: {best_fscore:.2f}%, Avg Kendall: {best_kendall:.4f}, Avg Spearman: {best_spearman:.4f}"
    )


if __name__ == "__main__":
    main()