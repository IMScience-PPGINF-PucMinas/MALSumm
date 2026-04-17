# -*- coding: utf-8 -*-
"""data_loader_cv.py — CV-aware version of data_loader.py.

Differences vs. the original:
  - VideoDataCV receives an explicit `keys` list instead of reading
    train/test keys from a split JSON by index.  The KFold partition
    is built once in main_cv.py and passed here directly.
  - Everything else is preserved: same HDF5 fields (features, gtscore),
    same __getitem__ contract (train -> features+gtscore,
    test -> features+video_name), same get_loader behaviour
    (DataLoader for train, raw Dataset for test).
  - 'both' mode is also supported.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

from configs.constants import (
    SUMME_DATASET_PATH,
    TVSUM_DATASET_PATH,
    MRHISUM_DATASET_PATH,
)


class VideoDataCV(Dataset):
    def __init__(self, mode: str, video_type: str, keys: list):
        """CV-aware Dataset: loads features and gtscores for an explicit key list.

        Args:
            mode       (str):  'train' or 'test'.
            video_type (str):  'SumMe', 'TVSum', 'MrHiSum', or 'both'.
            keys       (list): Video identifiers for this fold partition.
                               Built by KFold in main_cv.py — not read
                               from a split JSON here.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.keys = keys

        self.list_frame_features = []
        self.list_gtscores       = []

        if self.name == 'both':
            self._load_both()
        else:
            self._load(self._resolve_path())

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self) -> str:
        mapping = {
            'summe':   SUMME_DATASET_PATH,
            'tvsum':   TVSUM_DATASET_PATH,
            'mrhisum': MRHISUM_DATASET_PATH,
        }
        if self.name not in mapping:
            raise ValueError(f"Unsupported video_type: '{self.name}'")
        return mapping[self.name]

    # ------------------------------------------------------------------
    # Loading  — mirrors original VideoData.__init__ field names exactly
    # ------------------------------------------------------------------

    def _load(self, filename: str):
        """Load features and gtscores for self.keys from a single HDF5 file."""
        with h5py.File(filename, 'r') as hdf:
            for video_name in self.keys:
                self.list_frame_features.append(
                    torch.Tensor(np.array(hdf[f"{video_name}/features"]))
                )
                self.list_gtscores.append(
                    torch.Tensor(np.array(hdf[f"{video_name}/gtscore"]))
                )

    def _load_both(self):
        """Load from SumMe + TVSum for keys that belong to each file."""
        for path in (SUMME_DATASET_PATH, TVSUM_DATASET_PATH):
            with h5py.File(path, 'r') as hdf:
                for video_name in self.keys:
                    if video_name not in hdf:
                        continue          # key lives in the other file
                    self.list_frame_features.append(
                        torch.Tensor(np.array(hdf[f"{video_name}/features"]))
                    )
                    self.list_gtscores.append(
                        torch.Tensor(np.array(hdf[f"{video_name}/gtscore"]))
                    )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        """Train -> (frame_features, gtscore).  Test -> (frame_features, video_name)."""
        frame_features = self.list_frame_features[index]
        gtscore        = self.list_gtscores[index]

        if self.mode == 'test':
            return frame_features, self.keys[index]
        return frame_features, gtscore


# ---------------------------------------------------------------------------
# Public factory  (same contract as original get_loader, minus split_index)
# ---------------------------------------------------------------------------

def get_loader_cv(mode: str, video_type: str, keys: list):
    """Build a loader for an explicit list of video keys (one CV fold).

    Mirrors the original get_loader() behaviour exactly:
      - train -> DataLoader(dataset, batch_size=1, shuffle=True)
      - test  -> raw VideoDataCV  (Solver.evaluate iterates it directly)

    Args:
        mode       (str):  'train' or 'test'.
        video_type (str):  'SumMe', 'TVSum', 'MrHiSum', or 'both'.
        keys       (list): Video keys for this fold partition.

    Returns:
        DataLoader (train) or VideoDataCV (test).
    """
    dataset = VideoDataCV(mode, video_type, keys)

    if mode.lower() == 'train':
        return DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset      # same contract as original get_loader test path


if __name__ == '__main__':
    pass
