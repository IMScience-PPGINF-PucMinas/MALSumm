import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from configs.constants import *


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.datasets = [SUMME_DATASET_PATH,
                         TVSUM_DATASET_PATH,
                         MRHISUM_DATASET_PATH]
        self.splits_filename = 'data/splits/' + self.name + '_splits.json'
        self.split_index = split_index

        if self.name == 'both':
            self.splits_filenames = {
                'summe': SUMME_SPLIT_FILE_PATH,
                'tvsum': TVSUM_SPLIT_FILE_PATH
            }
            video_types = ['summe', 'tvsum']
            self.list_frame_features, self.list_gtscores = [], []

            for video_type in video_types:
                filename = self.datasets[video_type]
                splits_filename = self.splits_filenames[video_type]
                with open(splits_filename) as f:
                    data = json.loads(f.read())
                    for i, split in enumerate(data):
                        if i == self.split_index:
                            self.split = split
                            break

                    #split = data[self.split_index]
                    for video_name in self.split[self.mode + '_keys']:
                        with h5py.File(filename, 'r') as hdf:
                            frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
                            gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
                            print(f"Loaded video {video_name}: frame_features shape {frame_features.shape}, gtscore shape {gtscore.shape}")
                            self.list_frame_features.append(frame_features)
                            self.list_gtscores.append(gtscore)

        else:
            if self.name == 'summe':
                self.filename = self.datasets[0]
            elif self.name == 'tvsum':
                self.filename = self.datasets[1]
            elif self.name == 'mrhisum':
                self.filename = self.datasets[2]
            hdf = h5py.File(self.filename, 'r')
            self.list_frame_features, self.list_gtscores = [], []

            if self.name == 'mrhisum':
                with open(self.splits_filename) as f:
                    self.split = json.load(f)
            else:
                with open(self.splits_filename) as f:
                    data = json.loads(f.read())
                    for i, split in enumerate(data):
                        if i == self.split_index:
                            self.split = split
                            break

            for video_name in self.split[self.mode + '_keys']:
                frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
                gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))

                self.list_frame_features.append(frame_features)
                self.list_gtscores.append(gtscore)

        hdf.close()

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode + '_keys'])
        if self.name == 'mrhisum':
            return len(self.split[f'{self.mode}_keys'])
        else:
            return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.split[self.mode + '_keys'][index]
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]

        if self.mode == 'test':
            return frame_features, video_name
        else:
            return frame_features, gtscore


def get_loader(mode, video_type, split_index):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)


if __name__ == '__main__':
    pass
