import math

import numpy as np


def combine_dims(x):
    return np.concatenate((np.expand_dims(x[0::2], axis=1),
                           np.expand_dims(x[1::2], axis=1)), -1)


def repeat_dims(x, repeats=2):
    return np.repeat(np.expand_dims(x, axis=1), repeats, axis=1)


def khi(u1, u2, t):
    return 1 if (u1 <= t < u2) else 0


def omega(t, u0, u1):
    denom = u1 - u0
    return 0 if denom == 0 else (t - u0) / denom


def omegabar(t, u0, u1):
    return 1 - omega(t, u0, u1)


def bspline_n(i: int, k: int, t, u):
    if k == 1:
        return khi(u[i], u[i + 1], t)
    else:
        oa = omega(t, u[i], u[i + k - 1])
        ob = omegabar(t, u[i + 1], u[i + k])
        a = b = 0
        if 0 < oa <= 1:
            a = oa * bspline_n(i, k - 1, t, u)
        if 0 < ob <= 1:
            b = ob * bspline_n(i + 1, k - 1, t, u)
        # print("bs n ", oa, a, ob, b, "i", i, "k", k)
        return a + b
        # return omega(t, u[i], u[i + k - 1]) * bspline_n(i, k - 1, t, u) + omegabar(t, u[i + 1], u[i + k]) * bspline_n(i + 1, k - 1, t, u)


def lambda_neutre(a):
    return True


def lambda_not_neg(a):
    return np.logical_not(a < 0)


def lambda_0_to_1(a):
    return np.logical_and(0 < a, a < 1)


class TransformList:

    def __init__(self, transforms: list) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item


class CenterCoord:

    def __init__(self, items_idx: list = None):
        if items_idx is None:
            items_idx = ['vector_cp']
        self.items_idx = items_idx

    def __call__(self, item):
        for i in self.items_idx:
            item[i] -= .5
        return item


class RelativeCoordinate:

    def __init__(self, items_idx: list = None, keep_len: bool = False):
        if items_idx is None:
            items_idx = ['p']
        self.items_idx = items_idx
        self.keep_len = keep_len

    def __call__(self, item):
        for i in self.items_idx:
            item['abs_vector_' + i] = item['vector_' + i]
            item['vector_' + i] = item['vector_' + i][1:, :] - item['vector_' + i][:-1, :]
            if self.keep_len:
                item['vector_' + i] = np.pad(item['vector_' + i], ((1, 0), (0, 0)))
            else:
                item['mask_' + i] = item['mask_' + i][1:]
        return item


class Flip:

    def __init__(self, force=False):
        self.force = force

    def __call__(self, item):
        if self.force or item['rng'].integers(0, 1):
            u_max = item['debut_u'] + item['m'] + item['k']
            item['vector_u'][item['debut_u']:u_max] = (-np.flip(item['vector_u'][item['debut_u']:u_max])) + 1

            cp_max = (item['m'])
            item['vector_cp'][:cp_max, :] = np.flip(item['vector_cp'][:cp_max, :], 1)
        return item


class Rotation:

    def __init__(self, items_idx: list = None):
        if items_idx is None:
            items_idx = ['vector_cp']
        self.items_idx = items_idx

    def __call__(self, item):

        angle_rot = math.radians(item['rng'].integers(0, 360))
        cos_angle = math.cos(angle_rot)
        sin_angle = math.sin(angle_rot)
        item['angle_rot'] = angle_rot
        # print('idx ', item['idx'], ' seen ', item['seen'], 'angle ', angle_rot)
        d = item['d']
        ia = item['rng'].integers(0, d)
        ib = item['rng'].integers(0, d - 1)
        if ia == ib:
            ib = d - 1
        item['d_rot'] = [ia, ib]
        batch_diag = np.identity(d)
        batch_diag[ia, ia] = cos_angle
        batch_diag[ib, ib] = cos_angle
        batch_diag[ia, ib] = sin_angle
        batch_diag[ib, ia] = -sin_angle

        for i in self.items_idx:
            item[i] = np.matmul(item[i], batch_diag)

        return item


class Shifting:

    def __init__(self, shift_factor=0.5):
        self.shift_factor = shift_factor

    def __call__(self, item):
        precision = item['vector_ti'][1]
        n = item['m'] - item['k']
        for _, i in enumerate(item['rng'].integers(0, 3, 1 + int(item['seen'] * self.shift_factor))):
            if i == 0:
                # u
                u = item['k'] + item['debut_u'] + item['rng'].integers(0, n)
                a = max(item['vector_u'][u - 1] + precision, item['vector_u'][u] - 2 * precision)
                b = min(item['vector_u'][u + 1] - precision, item['vector_u'][u] + 2 * precision)
                if a < b:
                    item['vector_u'][u] = item['rng'].random() * (b - a) + a
                else:
                    item['vector_u'][u] = (a + b) / 2
            else:
                # cp
                cp = item['rng'].integers(0, item['m'])
                d = item['rng'].integers(0, item['d'])
                item['vector_cp'][cp, d] = item['vector_cp'][cp, d] + (item['rng'].random() * 0.1 - 0.05)

        return item


class RandomSizedTi:

    def __init__(self, size_factor: int = 2):
        self.size_factor = size_factor

    def __call__(self, item):
        taux_echant = item['rng'].integers(item['vector_ti'].size // self.size_factor, item['vector_ti'].size + 1)
        pas = 1. / (taux_echant - 1)

        item['vector_ti'] = np.arange(taux_echant) * pas
        item['mask_ti'] = item['mask_ti'][:taux_echant]
        return item


def make_vector_i_deb(item):
    target_length = len(item['vector_ti'])
    item['vector_i_deb'] = [0] * target_length
    k = item['k']
    u = item['vector_u'][item['debut_u']:]

    u_i_deb = 0
    u_i_fin = k
    _id = 0
    for i in range(target_length):
        ti = item['vector_ti'][i]
        while ti > u[u_i_fin]:
            u_i_fin += 1
            u_i_deb += 1

        item['vector_i_deb'][i] = u_i_deb
        _id = _id + 2

    return item


class Echantillonage:
    def __call__(self, item):
        target_length = len(item['vector_ti'])
        item['vector_p'] = np.zeros((target_length, item['d']))
        item['vector_i_deb'] = [0] * target_length
        k = item['k']
        m = item['m']
        u = item['vector_u'][item['debut_u']:]

        u_i_deb = 0
        u_i_fin = k
        _id = 0
        for i in range(target_length):
            ti = item['vector_ti'][i]
            while ti > u[u_i_fin]:
                u_i_fin += 1
                u_i_deb += 1

            item['vector_i_deb'][i] = u_i_deb
            if ti <= 0:
                item['vector_p'][_id, :] = item['vector_cp'][0, :]
            elif ti >= 1:
                item['vector_p'][_id, :] = item['vector_cp'][m - 1, :]
            else:
                for im in range(u_i_deb, u_i_deb + k):
                    n = bspline_n(im, k, ti, u)
                    if n != 0:
                        item['vector_p'][_id, :] += n * item['vector_cp'][im, :]
            _id = _id + 1

        item['mask_p'] = np.ones(item['vector_p'].shape[0])
        return item


class UMAP:

    def __init__(self, relative_output: bool = False, real_relative_output: bool = False):
        self.relative_output = relative_output
        self.real_relative_output = real_relative_output

    def __call__(self, item):
        if 'vector_i_deb' not in item:
            item = make_vector_i_deb(item)
        target_length = len(item['vector_ti'])
        k = item['k']
        m = item['m']
        u = item['vector_u'][item['debut_u']:]
        umap_len = 2 * k
        item['vector_u_map'] = np.zeros((target_length, umap_len))
        item['vector_u_neighbor'] = np.zeros((target_length, 2))
        for i in range(target_length):
            ti = item['vector_ti'][i]
            u_i_deb = item['vector_i_deb'][i]
            if i > 0 and u_i_deb != item['vector_i_deb'][i - 1]:
                item['vector_u_neighbor'][i][0] = u_i_deb - item['vector_i_deb'][i - 1]
                item['vector_u_neighbor'][i - 1][1] = u_i_deb - item['vector_i_deb'][i - 1]

            item['vector_u_map'][i] = u[u_i_deb: u_i_deb + umap_len]
            if self.relative_output:
                item['vector_u_map'][i, :k] += ti
                item['vector_u_map'][i, -k:] -= ti
            elif self.real_relative_output:
                item['vector_u_map'][i, :k] = ti - item['vector_u_map'][i, :k]
                item['vector_u_map'][i, -k:] -= ti

        return item


class InputNoise:

    def __init__(self, noise_ratio: float, noise_lvl: float, normal_noise: list = None,
                 items_idx: list = None, register_noise: bool = False):
        if items_idx is None:
            items_idx = ['p']
        self.register_noise = register_noise
        self.items_idx = items_idx
        self.normal_noise = normal_noise
        self.noise_lvl = noise_lvl
        self.noise_ratio = noise_ratio

    def __call__(self, item):
        rng = item['rng']
        if rng.random() <= self.noise_ratio:
            for i in self.items_idx:
                noise = 0
                if self.normal_noise is None:
                    noise = (rng.random(item['vector_' + i].shape) - .5) * self.noise_lvl
                    item['vector_' + i] = item['vector_' + i] + noise
                else:
                    item['vector_' + i] = item['vector_' + i] + (
                        rng.normal(self.normal_noise[0], self.normal_noise[1],
                                   item['vector_' + i].shape)) * self.noise_lvl
                if self.register_noise:
                    item['vector_noise_' + i] = noise
        elif self.register_noise:
            for i in self.items_idx:
                item['vector_noise_' + i] = np.zeros_like(item['vector_' + i])
        return item


class NormUnit:
    def __call__(self, item):
        p_min = np.min(item['vector_p'], axis=0)

        item['vector_p'] -= p_min
        item['vector_cp'] -= p_min

        p_max = np.max(item['vector_p'], axis=0)

        p_max = np.where(p_max < 1e-10, 1, p_max)

        item['norm_p'] = np.concatenate((np.expand_dims(p_min, 1), np.expand_dims(p_max, 1)), 1)

        item['vector_p'] /= p_max

        item['vector_cp'] /= p_max

        return item


class NormP:

    def __init__(self, input_list: list):
        self.input_list = input_list

    def normalize_p(self, item, norm_cp=False):

        p_mean = np.mean(item['vector_p'], axis=0)

        item['vector_p'] -= p_mean

        if norm_cp:
            item['vector_cp'] -= p_mean

        p_std = np.std(item['vector_p'], axis=0)
        p_std = np.where(p_std < 1e-10, 1, p_std)

        item['norm_p'] = np.concatenate((np.expand_dims(p_mean, 1), np.expand_dims(p_std, 1)), 1)

        item['vector_p'] /= p_std

        if norm_cp:
            item['vector_cp'] /= p_std

        if 'abs_vector_p' in item:
            abs_p_mean = np.mean(item['abs_vector_p'], axis=0)

            item['abs_vector_p'] -= abs_p_mean

            abs_p_std = np.std(item['abs_vector_p'], axis=0)
            abs_p_std = np.where(abs_p_std < 1e-10, 1, abs_p_std)

            item['abs_norm_p'] = np.concatenate((np.expand_dims(abs_p_mean, 1), np.expand_dims(abs_p_std, 1)), 1)

            item['abs_vector_p'] /= abs_p_std

        return item

    def __call__(self, item):
        if 'p' in self.input_list:
            item = self.normalize_p(item, 'cp' in self.input_list)
        return item


def pad_P(item: dict, target_length: int, time):
    if item['vector_p'].size < target_length * 2:
        pad = np.zeros((target_length * 2 - item['vector_p'].size))
        item['vector_p'] = np.concatenate((item['vector_p'], pad))
        item['mask_p'] = np.concatenate((item['mask_p'], pad))

        pad = np.zeros((target_length - item['vector_ti'].size))
        item['vector_ti'] = np.concatenate((item['vector_ti'], pad))
        item['mask_ti'] = np.concatenate((item['mask_ti'], pad))
        if time is not None:
            time = np.concatenate((time, pad))
    return item, time


def pad_vector(vector_, target_length: int, dim=0):
    pad_shape = list(vector_.shape)
    pad_shape[dim] = target_length - pad_shape[dim]
    if pad_shape[dim] > 0:
        return np.concatenate([vector_, np.zeros(pad_shape)], axis=dim)
    return vector_


def shared_token_prep(item: dict, input_list: list, time=None, rel_pos: bool = False,
                      keep_absolute_coord: bool = False):
    for i in input_list:
        if i in ['p', 'cp']:
            item['input_vector_' + i] = item['vector_' + i]

            if rel_pos:
                item['input_vector_' + i] = np.concatenate((item['input_vector_' + i], np.concatenate(
                    (item['input_vector_' + i][1:], item['input_vector_' + i][:1]))), -1)
                if keep_absolute_coord:
                    item['input_vector_' + i] = np.concatenate((item['input_vector_' + i],
                                                                item['abs_vector_' + i]), -1)
            a = item['input_mask_' + i] if ('input_mask_' + i) in item else item['mask_' + i]
            item['input_mask_' + i] = a
            if time is not None:
                item['input_vector_' + i] = np.concatenate((item['input_vector_' + i],
                                                            np.expand_dims(time, axis=-1)), -1)
        else:
            item['input_vector_' + i] = repeat_dims(item['vector_' + i],
                                                    item['d'] * (
                                                        2 if rel_pos else 1 + (1 if keep_absolute_coord else 0)) + (
                                                        1 if time is not None else 0))
    return item


class DataInputOutputPreping:
    def __init__(self, input_list: list, output_list: list, normalize: bool = False, softmax: bool = False,
                 shared_token: bool = False, add_time: bool = False, stack_u: int = 0, target_length: int = 128,
                 relative_coord_pos: bool = False, keep_absolute_coord: bool = False,
                 force_dim: int = -1):

        self.output_list = output_list
        self.norm = normalize
        self.normP = NormP(input_list)
        self.softmax = softmax
        self.shared_token = shared_token
        self.input_list = input_list
        self.add_time = add_time
        self.stack_u = stack_u
        self.target_length = target_length
        self.relative_coord_pos = relative_coord_pos
        self.keep_absolute_coord = keep_absolute_coord
        self.force_dim = force_dim
        if softmax and self.stack_u == 0:
            if 'u' in self.input_list:
                i = self.input_list.index('u')
                self.input_list.insert(i, 'softmax')
            else:
                self.input_list.append('softmax')

    def normalize_u(self, item):

        if self.softmax:
            ex_size = item['mask_u'].size
            n = item['m'] - item['k']
            size_u = 1 + ex_size - (np.count_nonzero(item['mask_u']) - n)
            new_u = np.zeros(size_u, dtype=float)
            new_input_mask = np.zeros(size_u, dtype=float)
            new_output_mask = np.zeros(size_u, dtype=float)
            pos_1 = item['debut_u'] + item['k']

            ex_input_mask = item['input_mask_u'] if 'input_mask_u' in item else item['mask_u']
            ex_output_mask = item['output_mask_u'] if 'output_mask_u' in item else item['mask_u']

            new_u[0] = item['vector_u'][pos_1]
            new_input_mask[0] = ex_input_mask[pos_1]
            new_output_mask[0] = ex_output_mask[pos_1]
            for i in range(1, n + 1):
                new_u[i] = item['vector_u'][pos_1 + i] - item['vector_u'][pos_1 + i - 1]
                new_input_mask[i] = ex_input_mask[pos_1 + i] * ex_input_mask[pos_1 + i - 1]
                new_output_mask[i] = 1 if ex_output_mask[pos_1 + i] or ex_output_mask[pos_1 + i - 1] else 0
            new_input_mask[n] = ex_input_mask[pos_1 + n - 1]
            item['vector_u'] = new_u
            item['input_mask_u'] = new_input_mask
            item['output_mask_u'] = new_output_mask
            item['mask_u'] = new_u > 0
            item['mask_u'] = item['mask_u'].astype(np.float32)

            item['vector_softmax'] = np.unpackbits(np.array([n + 1], dtype=np.uint8), axis=0)[3:]
            item['mask_softmax'] = np.ones_like(item['vector_softmax'])

        else:
            mask = np.logical_and(item['mask_u'], np.logical_not(0 > item['vector_u'])).astype(int)
            item['vector_u'] -= mask * 0.5

        return item

    def normalize_ti(self, item):
        item['vector_ti'] = item['vector_ti'][1:] - item['vector_ti'][:-1]
        item['mask_ti'] = item['mask_ti'][1:]
        return item

    def __call__(self, item):
        size_p = item['vector_p'].shape[0]
        time = None
        if self.add_time:
            time = item['vector_ti'][item['vector_ti'].size - size_p:] * 2 - 1

        item = self.normalize_u(item)
        item = self.normalize_ti(item)
        if self.norm:
            item = self.normP(item)

        target_dim = max(self.force_dim, item.get('max_d', -1))
        d = item['d']
        item['d'] = max(d, target_dim)
        list_dims = ['vector_p', 'vector_cp']
        if 'abs_vector_p' in item:
            list_dims.append('abs_vector_p')

        for key in list_dims:
            item[key] = pad_vector(item[key], target_dim, 1)

        if self.shared_token:
            item = shared_token_prep(item, input_list=self.input_list, time=time,
                                     rel_pos=self.relative_coord_pos, keep_absolute_coord=self.keep_absolute_coord)

        item['input_vector'] = np.concatenate(
            [item['input_vector_' + i] if ('input_vector_' + i) in item else item['vector_' + i]
             for i in self.input_list], axis=0)
        item['input_mask'] = np.concatenate(
            [item['input_mask_' + i] if ('input_mask_' + i) in item else item['mask_' + i]
             for i in self.input_list])
        input_len = item['input_vector'].shape[0]
        item['input_vector'] = pad_vector(item['input_vector'], target_length=self.target_length)
        item['input_mask'] = pad_vector(item['input_mask'], target_length=self.target_length)

        item['output_vector'] = []
        item['output_mask'] = []
        item['output_mask_softmax_net'] = []
        for i in self.output_list:
            if i == 'u' and self.stack_u > 0:
                for j in range(self.stack_u):
                    n = j + 1
                    if n == item['m'] - item['k']:
                        item['output_vector'].append(item['vector_' + i][:n + 1])
                        item['output_mask'].append(item['mask_' + i][:n + 1])
                        item['output_mask_softmax_net'].append(item['mask_' + i][:n + 1])
                    else:
                        item['output_vector'].append(np.ones(n + 1))
                        item['output_mask'].append(np.zeros(n + 1))
                        item['output_mask_softmax_net'].append(np.ones(n + 1))

            else:
                item['output_vector'].append(item['vector_' + i])
                item['output_mask'].append(
                    item['output_mask_' + i] if ('output_mask_' + i) in item else item['mask_' + i])

                if self.softmax:
                    item['output_mask_softmax_net'].append(
                        item['output_mask_' + i] if ('output_mask_' + i) in item else item['mask_' + i])

        output = {'input_vector': item['input_vector'],
                  'input_mask': item['input_mask'],
                  'output_vector': item['output_vector'],
                  'output_mask': item['output_mask'],
                  'output_mask_net': item['output_mask_softmax_net'] if self.softmax else item['output_mask'],
                  'idx': item['idx'],
                  'len': input_len,
                  }

        return output


class DataInputOutputPrepingUMAP:
    def __init__(self, input_list: list, output_list: list, normalize: bool = False,
                 add_time: bool = False, target_length: int = 128, relative_coord_pos: bool = False,
                 keep_absolute_coord: bool = False,
                 force_dim: int = -1):

        self.output_list = output_list
        self.norm = normalize
        self.normP = NormP(input_list)
        self.input_list = input_list
        self.add_time = add_time
        self.target_length = target_length
        self.relative_coord_pos = relative_coord_pos
        self.keep_absolute_coord = keep_absolute_coord
        self.force_dim = force_dim

    def __call__(self, item):
        size_p = item['vector_p'].shape[0]
        time = None
        if self.add_time:
            time = item['vector_ti'][item['vector_ti'].size - size_p:] * 2 - 1

        if self.norm:
            item = self.normP(item)

        # item, time = pad_P(item, self.target_length, time)
        target_dim = max(self.force_dim, item.get('max_d', -1))
        d = item['d']
        item['d'] = max(d, target_dim)
        list_dims = ['vector_p', 'vector_cp']
        if 'abs_vector_p' in item:
            list_dims.append('abs_vector_p')

        for key in list_dims:
            item[key] = pad_vector(item[key], target_dim, 1)

        item = shared_token_prep(item, input_list=self.input_list, time=time, rel_pos=self.relative_coord_pos,
                                 keep_absolute_coord=self.keep_absolute_coord)

        item['input_vector'] = np.concatenate(
            [item['input_vector_' + i] if ('input_vector_' + i) in item else item['vector_' + i]
             for i in self.input_list], axis=0)
        item['input_mask'] = np.concatenate(
            [item['input_mask_' + i] if ('input_mask_' + i) in item else item['mask_' + i]
             for i in self.input_list])

        item['vector_ti'] = np.expand_dims(np.array(item['vector_ti']), axis=1)
        item['output_vector'] = [pad_vector(item['vector_' + i], target_length=self.target_length)
                                 for i in self.output_list]
        input_len = item['input_vector'].shape[0]
        item['input_vector'] = pad_vector(item['input_vector'], target_length=self.target_length)
        item['input_mask'] = pad_vector(item['input_mask'], target_length=self.target_length)

        output = {
            'input_vector': item['input_vector'],
            'input_mask': item['input_mask'],
            'output_vector': item['output_vector'],
            'idx': item['idx'],
            'len': input_len,
        }

        return output


class DataMasking:

    def __init__(self, data_type, condition=lambda_neutre):
        self.data_type = data_type
        self.condition = condition

    def __call__(self, item):
        masked = np.logical_and(item['mask_' + self.data_type],
                                self.condition(item['vector_' + self.data_type])).astype(int)
        mask = np.logical_and(item['mask_' + self.data_type], np.logical_not(masked)).astype(int)
        item['input_mask_' + self.data_type] = mask
        item['output_mask_' + self.data_type] = masked
        return item


class TransformFactory:
    def __init__(self, relative_coord: bool, norm: bool, softmax: bool, shifting: bool,
                 echant: bool, shared_token: bool, add_time: bool, stack_u: int,
                 force_no_echant: bool, target_length: int, tl_len_factor: int,
                 noise_ratio: float, noise_lvl: float, normal_noise: list | None,
                 relative_output: bool, real_relative_output: bool, relative_coord_pos: bool,
                 norm_unit: bool, keep_absolute_coord: bool,
                 force_dim: int) -> None:
        super().__init__()
        self.relative_coord = relative_coord
        self.keep_absolute_coord = keep_absolute_coord
        self.norm = norm
        self.softmax = softmax
        self.shifting = shifting
        self.echant = echant
        self.shared_token = shared_token
        self.add_time = add_time
        self.stack_u = stack_u
        self.force_no_echant = force_no_echant
        self.condition = lambda_0_to_1 if self.softmax else lambda_neutre
        self.target_length = target_length
        self.tl_len_factor = tl_len_factor
        self.noise_ratio = noise_ratio
        self.noise_lvl = noise_lvl
        self.normal_noise = normal_noise
        self.relative_output = relative_output
        self.real_relative_output = real_relative_output
        self.relative_coord_pos = relative_coord_pos
        self.norm_unit = norm_unit
        self.force_dim = force_dim

    def factory_data_get(self, task: str):
        if task == 'u_map':
            return self.factory_data_get_u_map()
        else:
            return self.factory_data_get_default()

    def factory_data_get_default(self):
        masking = [DataMasking(data_type='u',
                               condition=self.condition if self.softmax else lambda_not_neg)]
        return self.generate_transform_list(['p'], ['u'], masking=masking)

    def factory_data_get_u_map(self):
        return self.generate_transform_list(['p'], ['u_map', 'ti'],
                                            masking=None, u_map=True)

    def generate_transform_list(self, input_types: list, output_types: list, masking: list = None,
                                u_map: bool = False, register_noise: bool = False):

        if u_map:
            inout = DataInputOutputPrepingUMAP(input_types, output_types, self.norm,
                                               add_time=self.add_time, target_length=self.target_length,
                                               relative_coord_pos=self.relative_coord_pos,
                                               keep_absolute_coord=self.keep_absolute_coord,
                                               force_dim=self.force_dim)
        else:
            inout = DataInputOutputPreping(input_types, output_types, self.norm, self.softmax, self.shared_token,
                                           add_time=self.add_time, stack_u=self.stack_u,
                                           target_length=self.target_length, relative_coord_pos=self.relative_coord_pos,
                                           keep_absolute_coord=self.keep_absolute_coord,
                                           force_dim=self.force_dim)

        transform_list = [[CenterCoord(), Flip()], [CenterCoord()]]

        if self.tl_len_factor > 1:
            for i in range(2):
                transform_list[i].append(RandomSizedTi(self.tl_len_factor))

        if self.shifting:
            transform_list[0].append(Shifting())

        transform_list[0].append(Rotation())

        if not self.force_no_echant and self.echant:
            for i in range(2):
                transform_list[i].append(Echantillonage())

        if u_map:
            for i in range(2):
                transform_list[i].append(UMAP(self.relative_output, self.real_relative_output))

        if self.norm_unit:
            for i in range(2):
                transform_list[i].append(NormUnit())

        if register_noise or (self.noise_ratio > 0. and self.noise_lvl > 0.):
            for i in range(2):
                transform_list[i].append(InputNoise(self.noise_ratio, self.noise_lvl, self.normal_noise,
                                                    register_noise=register_noise))
        if self.relative_coord or self.relative_coord_pos:
            for i in range(2):
                transform_list[i].append(RelativeCoordinate(keep_len=self.relative_coord_pos))

        if masking is not None:
            for i in range(2):
                transform_list[i].extend(masking)

        for i in range(2):
            transform_list[i].append(inout)

        return {'train': TransformList(transform_list[0]),
                'val': TransformList(transform_list[1])}


def factory_data_transforms(relative_coord: bool, norm: bool, softmax: bool, shifting: bool, echant: bool,
                            shared_token: bool, add_time: bool, stack_u: int,
                            target_length: int, tl_len_factor: int,
                            noise_ratio: float, noise_lvl: float, normal_noise: list | None,
                            task: str, relative_output: bool, real_relative_output: bool, relative_coord_pos: bool,
                            norm_unit: bool,
                            keep_absolute_coord: bool, force_dim: int,
                            force_no_echant: bool = False) -> dict:
    factory = TransformFactory(relative_coord=relative_coord, norm=norm, norm_unit=norm_unit, softmax=softmax,
                               shifting=shifting, echant=echant, shared_token=shared_token,
                               force_no_echant=force_no_echant, add_time=add_time, stack_u=stack_u,
                               target_length=target_length, tl_len_factor=tl_len_factor,
                               noise_ratio=noise_ratio, noise_lvl=noise_lvl, normal_noise=normal_noise,
                               relative_output=relative_output, real_relative_output=real_relative_output,
                               relative_coord_pos=relative_coord_pos,
                               keep_absolute_coord=keep_absolute_coord, force_dim=force_dim)
    return factory.factory_data_get(task)
