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
        return omega(t, u[i], u[i + k - 1]) * bspline_n(i, k - 1, t, u) + \
            omegabar(t, u[i + 1], u[i + k]) * bspline_n(i + 1, k - 1, t, u)


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

    def __init__(self, items_idx: list = None):
        if items_idx is None:
            items_idx = ['p']
        self.items_idx = items_idx

    def __call__(self, item):
        for i in self.items_idx:
            item['vector_' + i] = item['vector_' + i][2:] - item['vector_' + i][:-2]
            item['mask_' + i] = item['mask_' + i][2:]
        return item


class Flip:

    def __init__(self, force=False):
        self.force = force

    def __call__(self, item):
        if self.force or item['rng'].integers(0, 1):
            u_max = item['debut_u'] + item['m'] + item['k']
            item['vector_u'][item['debut_u']:u_max] = (-np.flip(item['vector_u'][item['debut_u']:u_max])) + 1

            cp_max = (item['m']) * 2
            item['vector_cp'][:cp_max:2] = np.flip(item['vector_cp'][:cp_max:2])
            item['vector_cp'][1:cp_max:2] = np.flip(item['vector_cp'][1:cp_max:2])
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

        for i in self.items_idx:
            seq_size = np.size(item[i], 0)
            batch_diag = np.zeros((seq_size, seq_size))

            for p in range(0, seq_size, 2):
                batch_diag[p][p] = cos_angle
                batch_diag[p][p + 1] = sin_angle
                batch_diag[p + 1][p] = -sin_angle
                batch_diag[p + 1][p + 1] = cos_angle

            item[i] = np.matmul(item[i], batch_diag)

        return item


class Shifting:

    def __init__(self, shift_factor=0.5):
        self.shift_factor = shift_factor

    def __call__(self, item):
        precision = item['vector_ti'][1]
        n = item['m'] - item['k']
        # print(item['seen'])
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
                cp = item['rng'].integers(0, item['m'] * 2)
                item['vector_cp'][cp] = item['vector_cp'][cp] + (item['rng'].random() * 0.1 - 0.05)

        return item


class Echantillonage:
    def __call__(self, item):
        target_length = len(item['vector_ti'])
        item['vector_p'] = np.zeros(target_length * 2)
        k = item['k']
        m = item['m']
        for i in range(0, target_length):
            ti = item['vector_ti'][i]
            for j in range(2):
                id_ = i * 2 + j
                if ti <= 0:
                    item['vector_p'][id_] = item['vector_cp'][j]
                elif ti >= 1:
                    item['vector_p'][id_] = item['vector_cp'][m * 2 - 2 + j]
                else:
                    for im in range(m):
                        item['vector_p'][id_] += bspline_n(im, k, ti, item['vector_u'][item['debut_u']:]) \
                                                 * item['vector_cp'][im * 2 + j]

        item['mask_p'] = np.ones_like(item['vector_p'])

        return item


class NormP:

    def __init__(self, input_list: list):
        self.input_list = input_list

    def normalize_p(self, item, norm_cp=False):
        mask_px = np.zeros_like(item['vector_p'], dtype=int)
        mask_px[::2] = 1
        mask_cpx = np.zeros_like(item['vector_cp'], dtype=int)
        mask_cpx[::2] = 1

        mean_x = np.mean(item['vector_p'], where=mask_px.astype(bool))
        mean_y = np.mean(item['vector_p'], where=np.logical_not(mask_px))

        item['vector_p'] -= mask_px * mean_x
        item['vector_p'] -= (1 - mask_px) * mean_y

        if norm_cp:
            item['vector_cp'] -= item['mask_cp'] * mask_cpx * mean_x
            item['vector_cp'] -= item['mask_cp'] * (1 - mask_cpx) * mean_y

        std_x = np.std(item['vector_p'], where=mask_px.astype(bool))
        std_y = np.std(item['vector_p'], where=np.logical_not(mask_px))

        item['vector_p'] /= mask_px * std_x + (1 - mask_px) * std_y

        if norm_cp:
            item['vector_cp'] /= (1 - item['mask_cp']) + \
                                 item['mask_cp'] * mask_cpx * std_x \
                                 + item['mask_cp'] * (1 - mask_cpx) * std_y
        return item

    def __call__(self, item):
        if 'p' in self.input_list:
            item = self.normalize_p(item, 'cp' in self.input_list)
        return item


class DataInputOutputPreping:
    def __init__(self, input_list: list, output_list: list, normalize: bool = False, softmax: bool = False,
                 shared_token: bool = False, add_time: bool = False, stack_u: int = 0, target_length: int = 128):

        self.output_list = output_list
        self.norm = normalize
        self.normP = NormP(input_list)
        self.softmax = softmax
        self.shared_token = shared_token
        self.input_list = input_list
        self.add_time = add_time
        self.stack_u = stack_u
        self.target_length = target_length
        if softmax and self.stack_u == 0:
            if 'u' in self.input_list:
                i = self.input_list.index('u')
                self.input_list.insert(i, 'softmax')
            else:
                self.input_list.append('softmax')

    def normalize_u(self, item):

        if self.softmax:
            # mask = np.logical_and(item['mask_u'], np.logical_and(0 < item['vector_u'],
            # 1 > item['vector_u'])).astype(int)
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
        # np.mean(item['vector_u'], where=mask.astype(bool))
        # item['vector_u'] /= mask * np.std(item['vector_u'], where=mask.astype(bool))

        return item

    def normalize_ti(self, item):
        item['vector_ti'] = item['vector_ti'][1:] - item['vector_ti'][:-1]
        item['mask_ti'] = item['mask_ti'][1:]
        return item

    def __call__(self, item):
        size_p = item['vector_p'].size // 2
        if self.add_time:
            time = item['vector_ti'][item['vector_ti'].size - size_p:] * 2 - 1

        item = self.normalize_u(item)
        item = self.normalize_ti(item)
        if self.norm:
            item = self.normP(item)

        if item['vector_p'].size < self.target_length * 2:
            pad = np.zeros((self.target_length * 2 - item['vector_p'].size))
            item['vector_p'] = np.concatenate((item['vector_p'], pad))
            item['mask_p'] = np.concatenate((item['mask_p'], pad))

            pad = np.zeros((self.target_length - item['vector_ti'].size))
            item['vector_ti'] = np.concatenate((item['vector_ti'], pad))
            item['mask_ti'] = np.concatenate((item['mask_ti'], pad))
            if self.add_time:
                time = np.concatenate((time, pad))

        if self.shared_token:
            for i in self.input_list:
                if i in ['p', 'cp']:
                    item['input_vector_' + i] = combine_dims(item['vector_' + i])
                    a = item['input_mask_' + i] if ('input_mask_' + i) in item else item['mask_' + i]
                    item['input_mask_' + i] = a[0::2]
                    if self.add_time:
                        item['input_vector_' + i] = np.concatenate((item['input_vector_' + i],
                                                                    np.expand_dims(time, axis=1)), -1)
                else:
                    item['input_vector_' + i] = repeat_dims(item['vector_' + i], 3 if self.add_time else 2)

        item['input_vector'] = np.concatenate(
            [item['input_vector_' + i] if ('input_vector_' + i) in item else item['vector_' + i]
             for i in self.input_list], axis=0)
        item['input_mask'] = np.concatenate(
            [item['input_mask_' + i] if ('input_mask_' + i) in item else item['mask_' + i]
             for i in self.input_list])

        item['output_vector'] = []
        item['output_mask'] = []
        item['output_mask_softmax_net'] = []
        # item['loss_factor'] = []
        for i in self.output_list:
            if i == 'u' and self.stack_u > 0:
                for j in range(self.stack_u):
                    n = j + 1
                    if n == item['m'] - item['k']:
                        # item['loss_factor'].append(1.)
                        item['output_vector'].append(item['vector_' + i][:n + 1])
                        item['output_mask'].append(item['mask_' + i][:n + 1])
                        item['output_mask_softmax_net'].append(item['mask_' + i][:n + 1])
                    else:
                        # item['loss_factor'].append(0.)
                        item['output_vector'].append(np.ones(n + 1))
                        item['output_mask'].append(np.zeros(n + 1))
                        item['output_mask_softmax_net'].append(np.ones(n + 1))

            else:
                item['output_vector'].append(item['vector_' + i])
                item['output_mask'].append(
                    item['output_mask_' + i] if ('output_mask_' + i) in item else item['mask_' + i])
                # item['loss_factor'].append(1.)

                if self.softmax:
                    item['output_mask_softmax_net'].append(
                        item['output_mask_' + i] if ('output_mask_' + i) in item else item['mask_' + i])

            return {'input_vector': item['input_vector'],
                    'input_mask': item['input_mask'],
                    'output_vector': item['output_vector'],
                    'output_mask': item['output_mask'],
                    'output_mask_net': item['output_mask_softmax_net'] if self.softmax else item['output_mask'],
                    # 'loss_factor': item['loss_factor'],
                    'idx': item['idx']
                    }


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
                 force_no_echant: bool, target_length: int) -> None:
        super().__init__()
        self.relative_coord = relative_coord
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

    def factory_data_get(self):
        masking = [DataMasking(data_type='u',
                               condition=self.condition if self.softmax else lambda_not_neg)]
        return self.generate_transform_list(['p'], ['u'], masking=masking)

    def generate_transform_list(self, input_types: list, output_types: list, masking: list = None):

        inout = DataInputOutputPreping(input_types, output_types, self.norm, self.softmax, self.shared_token,
                                       add_time=self.add_time, stack_u=self.stack_u,
                                       target_length=self.target_length)
        transform_list = [[CenterCoord(), Flip()],
                          [CenterCoord()]]

        if self.shifting:
            transform_list[0].append(Shifting())

        transform_list[0].append(Rotation())

        if not self.force_no_echant and self.echant:
            for i in range(2):
                transform_list[i].append(Echantillonage())

        if self.relative_coord:
            for i in range(2):
                transform_list[i].append(RelativeCoordinate())

        if masking is not None:
            for i in range(2):
                transform_list[i].extend(masking)

        for i in range(2):
            transform_list[i].append(inout)

        return {'train': TransformList(transform_list[0]),
                'val': TransformList(transform_list[1])}


def factory_data_transforms(relative_coord: bool, norm: bool, softmax: bool, shifting: bool, echant: bool,
                            shared_token: bool, add_time: bool, stack_u: int = 0, force_no_echant: bool = False,
                            target_length: int = 128) -> dict:
    factory = TransformFactory(relative_coord=relative_coord, norm=norm, softmax=softmax,
                               shifting=shifting, echant=echant, shared_token=shared_token,
                               force_no_echant=force_no_echant, add_time=add_time, stack_u=stack_u,
                               target_length=target_length)
    return factory.factory_data_get()
