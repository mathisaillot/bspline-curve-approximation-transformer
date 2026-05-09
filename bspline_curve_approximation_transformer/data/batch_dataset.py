import random

import numpy as np
import torch

from bspline_curve_approximation_transformer.data import BSplineDataset


class BatchDataset(BSplineDataset):
    def __init__(self, data_bspline: np.ndarray,
                 k: int = 4, m: int = 1, target_length: int = 128,
                 segment: bool = False, n_segment: int = 1, segment_part: int = 1):
        super().__init__()

        self.target_length = target_length
        if target_length > 0:
            self.vector_ti = np.array([i / (self.target_length - 1) for i in range(self.target_length)])

        self.m = m
        self.k = k

        self.length = data_bspline.shape[0]
        self.d = data_bspline.shape[2]

        if segment:
            assert n_segment >= segment_part
            n = data_bspline.shape[1] - 1
            a = round((segment_part - 1) * n / n_segment)
            b = round(segment_part * n / n_segment)

            self.n = b - a + 1

            #print(a, b, self.n)

            #data_p = np.zeros((self.length, self.n, self.d))
            data_p = data_bspline[:, a:b + 1, :]

        else:
            self.n = data_bspline.shape[1]
            data_p = data_bspline

        if self.n != self.target_length:
            if target_length > 0:
                self.data_p = np.zeros((self.length, self.target_length, self.d))
                vector_ti = np.array([i / (self.n - 1) for i in range(self.n)])
                j = 0
                tj = vector_ti[j]
                for i in range(self.target_length):
                    ti = self.vector_ti[i]
                    while tj < ti:
                        j += 1
                        tj = vector_ti[j]

                    if ti == tj:
                        self.data_p[:, i, :] = data_p[:, j, :]
                    else:
                        j1 = j - 1
                        tj1 = vector_ti[j1]
                        alpha = (tj - ti) / (tj - tj1)
                        self.data_p[:, i, :] = (data_p[:, j1, :] * alpha +
                                                     (1 - alpha) * data_p[:, j, :])
            else:
                self.target_length = self.n
                self.vector_ti = np.array([i / (self.n - 1) for i in range(self.n)])
                self.data_p = data_p
        else:
            self.data_p = data_p

        self.max_m = m

        self.nodes_data_length_max = self.max_m + k + k
        self.control_point_data_length_max = self.max_m + k
        self.reset()

        fake_u = np.zeros(m + k + k)
        fake_u[k:k + m] = 0.1 * np.arange(1, m + 1, 1) if m > 1 else 0.5
        fake_u[k + m:] = 1

        # print(fake_u)

        total_pad = (self.max_m + 2 * k) - len(fake_u)
        pad_left = random.randint(0, total_pad)

        self.vector_u = np.pad(np.pad(fake_u, (1, 1), 'constant', constant_values=(-1, 1.5)),
                               (pad_left, total_pad - pad_left), 'constant')
        self.mask_u = np.pad(np.ones(len(fake_u) + 2), (pad_left, total_pad - pad_left), 'constant')
        self.debut_u = pad_left + 1


    def __get_nodes_data_lentgh_max__(self):
        return self.nodes_data_length_max

    def __get_control_point_data_length_max__(self):
        return self.control_point_data_length_max

    def __gettargetlength__(self):
        return self.target_length

    def reset(self, epoch_number=0):
        self.seen = np.repeat(epoch_number, self.__len__())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.seen[idx] = self.seen[idx] + 1
        n = self.seen[idx]
        rng = np.random.default_rng(seed=random.randint(100000, 100000000))

        #size_u = self.m + self.k + self.k

        item = {'vector_ti': self.vector_ti,
                'vector_p': self.data_p[idx],
                'vector_cp': np.ones((self.m, self.d)),
                'vector_u': self.vector_u,
                'mask_u': self.mask_u,
                'debut_u': self.debut_u,
                'idx': idx, 'k': self.k,
                'm': self.m + self.k,
                'd': self.d,
                'rng': rng, 'seen': n}
        item['mask_ti'] = np.ones_like(item['vector_ti'])
        item['mask_p'] = np.ones_like(item['vector_p'][:,0])
        return item