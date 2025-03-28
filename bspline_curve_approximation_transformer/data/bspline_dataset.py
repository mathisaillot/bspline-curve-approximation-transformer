import os
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset


def get_file_info_dataset() -> str:
    return "dataset_info.csv"


def get_train_idx_dataset() -> str:
    return "train_idx.txt"


def get_val_idx_dataset() -> str:
    return "val_idx.txt"


class BSplineDataset(Dataset):

    def __init__(self):
        super().__init__()

    def __get_nodes_data_lentgh_max__(self):
        return 0

    def __get_control_point_data_length_max__(self):
        return 0

    def __gettargetlength__(self):
        return 0

    def get_train_idx(self) -> list:
        return []

    def get_val_idx(self) -> list:
        return []

    def reset(self, epoch_number=0):
        return

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None


class BSplineSequenceDataset(BSplineDataset):

    def __init__(self, target_length: int, root_dir: str = "bspline_dataset",
                 min_nodes_length: int = 0, mini_seed: int = 666, padding_u: bool = False):

        super().__init__()
        self.root_dir = root_dir
        self.target_length = target_length
        self.mini_seed = mini_seed % 250000
        self.vector_ti = np.array([i / (self.target_length - 1) for i in range(self.target_length)])
        info_file = os.path.join(self.root_dir, get_file_info_dataset())
        data_info = pd.read_csv(info_file, header=0, index_col=False)

        self.train_idx = pd.read_csv(os.path.join(self.root_dir, "train_idx.txt"),
                                     header=None, index_col=False).to_numpy()
        self.val_idx = pd.read_csv(os.path.join(self.root_dir, "val_idx.txt"),
                                   header=None, index_col=False).to_numpy()
        self.nodes_data = list()
        self.k_data = list()
        self.m_data = list()
        self.control_point_data = list()
        self.padding_u = padding_u
        self.nodes_data_length_max = min_nodes_length
        self.control_point_data_length_max = 0
        self.seen = None
        curve_dir = os.path.join(self.root_dir, 'curves')

        for i, file in enumerate(data_info['File Name']):
            file_path = os.path.join(curve_dir, file)
            data_bspline = pd.read_csv(file_path, header=0, index_col=False).to_numpy()[:, 1:]

            k = int(data_info['Order'][i])
            n = int(data_info['Internal Knots'][i])
            n_courbes = int(data_info['Count'][i])
            n_nodes = k * 2 + n
            n_control_points = 2 * (n + k)
            self.k_data.extend([k] * n_courbes)
            self.m_data.extend([n + k] * n_courbes)
            self.nodes_data += data_bspline[:, :n_nodes].tolist()
            self.control_point_data += data_bspline[:, n_nodes:].tolist()
            self.nodes_data_length_max = max(self.nodes_data_length_max, n_nodes)
            self.control_point_data_length_max = max(self.control_point_data_length_max, n_control_points)
        self.length = len(self.nodes_data)
        self.reset()

    def __get_nodes_data_lentgh_max__(self):
        return self.nodes_data_length_max

    def __get_control_point_data_length_max__(self):
        return self.control_point_data_length_max

    def __gettargetlength__(self):
        return self.target_length

    def reset(self, epoch_number=0):
        self.seen = np.repeat(epoch_number, self.__len__())

    def get_train_idx(self) -> list:
        return self.train_idx

    def get_val_idx(self) -> list:
        return self.val_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        self.seen[idx] = self.seen[idx] + 1
        n = self.seen[idx]
        rng = np.random.default_rng(seed=random.randint(100000, 100000000))

        item = {'vector_ti': self.vector_ti,
                'idx': idx, 'k': self.k_data[idx],
                'm': self.m_data[idx],
                'rng': rng, 'seen': n}
        item['mask_ti'] = np.ones_like(item['vector_ti'])

        param_type = 'nodes'

        i = self.__getattribute__(param_type + '_data')[idx]

        if self.padding_u:
            total_pad = self.__getattribute__(param_type + '_data_length_max') - len(i)
            pad_left = 0

            item['vector_u'] = np.pad(i, (pad_left, total_pad - pad_left), 'constant')
            item['mask_u'] = np.pad(np.ones(len(i)), (pad_left, total_pad - pad_left), 'constant')
            item['debut_u'] = pad_left
        else:
            item['vector_u'] = np.array(i)
            item['mask_u'] = np.ones(len(i))
            item['debut_u'] = 0

        param_type = 'control_point'

        i = self.__getattribute__(param_type + '_data')[idx]
        item['vector_cp'] = np.pad(i, (0, self.__getattribute__(param_type + '_data_length_max') - len(i)), 'constant')

        item['mask_cp'] = np.pad(np.ones(len(i)), (0, self.__getattribute__(param_type + '_data_length_max') - len(i)),
                                 'constant')
        # item['debut_cp'] = 0

        return item


class BSplineSequenceDatasetWithTransform(BSplineDataset):

    def __init__(self, dataset: BSplineDataset, transform: callable) -> None:
        super().__init__()
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index):
        return self._transform(self._dataset.__getitem__(index))

    def __get_nodes_data_lentgh_max__(self):
        return self._dataset.__get_nodes_data_lentgh_max__()

    def __get_control_point_data_length_max__(self):
        return self._dataset.__get_control_point_data_length_max__()

    def get_train_idx(self) -> list:
        return self._dataset.get_train_idx()

    def get_val_idx(self) -> list:
        return self._dataset.get_val_idx()

    def __gettargetlength__(self):
        return self._dataset.__gettargetlength__()

    def reset(self, epoch_number=0):
        self._dataset.reset(epoch_number)

    def __len__(self):
        return self._dataset.__len__()
