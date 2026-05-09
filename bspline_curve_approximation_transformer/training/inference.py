import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from bspline_curve_approximation_transformer.data import BSplineSequenceDatasetWithTransform, factory_data_transforms, \
    DataPretreator, PARNETDataset, BatchDataset
from bspline_curve_approximation_transformer.transformer import CustomTransformer, UMAPTransformer
from bspline_curve_approximation_transformer.utils import LoggerCSV


def custom_inference(
        data_bspline: np.ndarray,
        k: int,
        m: int,
        n_segment: int,
        device: torch.device,
        dataset_root_dir: str,
        target_length: int,
        tl_len_factor: int,
        dataset_min_nodes: int,
        norm: bool,
        norm_unit: bool,
        output_folder: str,
        random_seed: int,
        shuffle_seed: int,
        relative_coord: bool,
        relative_coord_pos: bool,
        keep_absolute_coord: bool,
        relative_output: bool,
        real_relative_output: bool,
        epochs: int,
        batch_size: int,
        hidden_size: int,
        n_heads: int,
        depth: int,
        dim_head: int,
        model_type: str,
        softmax: bool,
        adam: bool,
        shifting: bool,
        shared_token: bool,
        learnable_pe: bool,
        no_pe: bool,
        residual: bool,
        attn_dropout: float,
        ff_dropout: float,
        add_time: bool,
        scale_loss: bool,
        loss_scaling: list | int,
        stack_u: int,
        force_dim: int,
        noise_ratio: float,
        noise_lvl: float,
        normal_noise: list | None,
        lr: float,
        momentum: float,
        gamma: float,
        step: int,
        warmup_epochs: int,
        snapshot_frequency: int,
        task: str,
        training_name: str = "BsplineDefaultParameters",
        total_epochs: int = 0,
        workers_train: int = 0,
        workers_val: int = 0,
        load_snapshot: bool = False,
        model_parameters=None,
        optimizer_cache=None,
        scheduler_cache=None,
        starting_epoch=0,
        fast_test: bool = False,
        verbose_log: bool = True,
        eval_only: bool = False,
        snap: str = "",
        backbone_snapshot: str = None,
        freeze_backbone: bool = False,
        lastsnap: str = None):
    assert n_segment > 0

    segment = n_segment > 1

    torch.cuda.empty_cache()
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    print(f"STARTING ON DEVICE : {device}")

    target_dir = os.path.join(output_folder, training_name)
    os.makedirs(target_dir, exist_ok=True)

    target_file = os.path.join(target_dir, f"output_{os.path.split(snap)[1]}_m{m}_k{k}_n_seg{n_segment}.csv")

    print(f"OUTPUT PATH {target_file}")
    loggers_output = LoggerCSV(target_file)
    entete = ["id", "knots..."]
    loggers_output.log(entete, False)

    transforms = factory_data_transforms(relative_coord=relative_coord, norm=norm, softmax=softmax,
                                         shifting=False, echant=False, shared_token=shared_token,
                                         add_time=add_time, stack_u=stack_u, target_length=0,
                                         tl_len_factor=1,
                                         noise_lvl=0., noise_ratio=0., normal_noise=None,
                                         task=task, relative_output=relative_output,
                                         real_relative_output=real_relative_output,
                                         keep_absolute_coord=keep_absolute_coord,
                                         relative_coord_pos=relative_coord_pos, force_dim=force_dim,
                                         norm_unit=norm_unit)
    model = None
    output_knot_vectors = []

    for part in range(n_segment):
        segment_part = part + 1
        dataset = BatchDataset(data_bspline, k=k, m=m, target_length=target_length,
                               segment=segment, n_segment=n_segment, segment_part=segment_part)

        dataset_eval = BSplineSequenceDatasetWithTransform(dataset, transform=transforms['val'])

        datatreatment = DataPretreator(device=device, dataset=dataset_eval)

        dataset_idx = np.arange(len(dataset))

        input_size = datatreatment.get_input_length()
        input_depth = datatreatment.get_input_depth()

        val_loader = DataLoader(dataset_eval, batch_size=batch_size, sampler=dataset_idx, num_workers=workers_val)
        if model is None:
            if task == 'u_map':
                model = UMAPTransformer(
                    hidden_size=hidden_size,
                    dim_head=dim_head,
                    n_heads=n_heads,
                    depth=depth,
                    max_seq_len=input_size,
                    input_depth=input_depth,
                    output_depth=datatreatment.get_output_length(),
                    func_out='ReLU',
                    learnable_pe=learnable_pe,
                    no_pe=no_pe,
                    residual=residual,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                )
            else:
                assert 1 <= m <= len(datatreatment.get_output_length())
                model = CustomTransformer(
                    hidden_size=hidden_size,
                    dim_head=dim_head,
                    n_heads=n_heads,
                    depth=depth,
                    max_seq_len=input_size,
                    input_depth=input_depth,
                    n_class=datatreatment.get_output_length(),
                    softmax=softmax,
                    learnable_pe=learnable_pe,
                    network_type=model_type,
                    no_pe=no_pe,
                    residual=residual,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                )

            model.load_state_dict(model_parameters)

            model = model.float()
            model.to(device)

            model.eval()

        total_iteration = len(val_loader)
        iteration = 1

        for item in val_loader:
            vector_batch = []
            (input_vector, input_mask, input_len,
             output_vector, output_mask, output_mask_net) = datatreatment.data_treatment(item)
            with ((torch.set_grad_enabled(False))):
                if task == 'u_map':
                    outputs = model(src=input_vector, src_mask=input_mask, input_len=input_len)
                    # print("res ", outputs[0])
                    for i in range(outputs[0].size(0)):
                        log_line = [0, item['idx'][i].item()]

                        for k in range(outputs[0].size(1)):
                            log_line += [input_mask[i, k].int().item()]
                            for j, o in enumerate(outputs):
                                log_line += o[i, k, :].tolist() + output_vector[j][i, k, :].tolist()
                        loggers_output.log(log_line, False)
                    loggers_output.save_log()
                else:
                    outputs = model(src=input_vector, src_mask=input_mask,
                                    output_mask=output_mask_net, input_len=input_len)
                    # print("res ", outputs[0])
                    u = m - 1
                    for i, out in enumerate(outputs[u]):
                        knot_vector = [0. for _ in range(k)]
                        for knot in out.tolist():
                            knot_vector.append(knot_vector[-1] + knot)

                        knot_vector = [elem / knot_vector[-1] for elem in knot_vector]
                        knot_vector[-1] = 1.

                        for j in range(k - 1):
                            knot_vector.append(1.)

                        vector_batch.append([int(item['idx'][i].item())] + knot_vector)

            if segment:
                u_divs = 1. / n_segment
                u_dec = u_divs * (segment_part - 1)

                for vec in vector_batch:
                    if segment_part == 1:
                        output_knot_vectors.append(vec[:k + 1])
                    id_vec = vec[0]
                    if segment_part > 1:
                        output_knot_vectors[id_vec].append(u_dec)
                    for kn in vec[k+1:-k]:
                        output_knot_vectors[id_vec].append(u_dec + kn * u_divs)
                    if segment_part == n_segment:
                        output_knot_vectors[id_vec] += [1. for _ in range(k)]

            else:
                output_knot_vectors += vector_batch
            print(f"Iteration {iteration}/{total_iteration}")
            iteration += 1
    print("Save all results")
    for log_line in output_knot_vectors:
        loggers_output.log(log_line, False)
    loggers_output.save_log()
    del model
    torch.cuda.empty_cache()

    return output_knot_vectors


def parnet_inference(k: int,
                     m: int,
                     set_name: str,
                     segment: bool,
                     n_segment: int,
                     segment_part: int,
                     output_length: int,
                     device: torch.device,
                     dataset_root_dir: str,
                     target_length: int,
                     tl_len_factor: int,
                     dataset_min_nodes: int,
                     norm: bool,
                     norm_unit: bool,
                     output_folder: str,
                     random_seed: int,
                     shuffle_seed: int,
                     relative_coord: bool,
                     relative_coord_pos: bool,
                     relative_output: bool,
                     real_relative_output: bool,
                     keep_absolute_coord: bool,
                     epochs: int,
                     batch_size: int,
                     hidden_size: int,
                     n_heads: int,
                     depth: int,
                     dim_head: int,
                     model_type: str,
                     softmax: bool,
                     adam: bool,
                     shifting: bool,
                     shared_token: bool,
                     learnable_pe: bool,
                     no_pe: bool,
                     residual: bool,
                     attn_dropout: float,
                     ff_dropout: float,
                     add_time: bool,
                     scale_loss: bool,
                     loss_scaling: list,
                     stack_u: int,
                     noise_ratio: float,
                     noise_lvl: float,
                     normal_noise: list | None,
                     force_dim: int,
                     lr: float,
                     momentum: float,
                     gamma: float,
                     step: int,
                     warmup_epochs: int,
                     snapshot_frequency: int,
                     task: str,
                     training_name: str = "BsplineDefaultParameters",
                     total_epochs: int = 0,
                     # workers_train: int = 0,
                     workers_val: int = 0,
                     load_snapshot: bool = False,
                     model_parameters=None,
                     optimizer_cache=None,
                     scheduler_cache=None,
                     starting_epoch=0,
                     backbone_snapshot: str = None,
                     freeze_backbone: bool = False,
                     snap: str = ""):
    torch.cuda.empty_cache()
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    print(f"STARTING ON DEVICE : {device}")

    # todo parnet for not stack u

    if task == 'u_map':
        directory = f"k{k}_n_seg{n_segment}_seg_part{segment_part}_out{output_length}"

        target_dir = os.path.join(output_folder, 'parnet_results', training_name, set_name, directory)
        os.makedirs(target_dir, exist_ok=True)

        target_file = os.path.join(target_dir, 'output.csv')

        print(f"OUTPUT PATH {target_file}")
        loggers_output = LoggerCSV(target_file)
    else:
        loggers_output = []
        for i in range(stack_u):
            directory = f"m{i + 1}_k{k}_n_seg{n_segment}_seg_part{segment_part}_out{output_length}"

            target_dir = os.path.join(output_folder, 'parnet_results', training_name, set_name, directory)
            os.makedirs(target_dir, exist_ok=True)

            target_file = os.path.join(target_dir, 'output.csv')

            print(f"OUTPUT PATH {target_file}")
            loggers_output.append(LoggerCSV(target_file))

    dataset = PARNETDataset(set_name=set_name, k=k, m=m, target_length=output_length,
                            segment=segment, n_segment=n_segment, segment_part=segment_part)

    # logger_output = LoggerCSV(target_file)

    transforms = factory_data_transforms(relative_coord=relative_coord, norm=norm, softmax=softmax,
                                         shifting=False, echant=False, shared_token=shared_token,
                                         add_time=add_time, stack_u=stack_u, target_length=0,
                                         tl_len_factor=1,
                                         noise_lvl=0., noise_ratio=0., normal_noise=None,
                                         task=task, relative_output=relative_output,
                                         real_relative_output=real_relative_output,
                                         keep_absolute_coord=keep_absolute_coord,
                                         relative_coord_pos=relative_coord_pos, force_dim=force_dim,
                                         norm_unit=norm_unit)

    dataset_eval = BSplineSequenceDatasetWithTransform(dataset, transform=transforms['val'])

    datatreatment = DataPretreator(device=device, dataset=dataset_eval)

    dataset_idx = np.arange(len(dataset))

    input_size = datatreatment.get_input_length()
    input_depth = datatreatment.get_input_depth()

    val_loader = DataLoader(dataset_eval, batch_size=batch_size, sampler=dataset_idx, num_workers=workers_val)

    epsilon = 1e-5
    if task == 'u_map':
        model = UMAPTransformer(
            hidden_size=hidden_size,
            dim_head=dim_head,
            n_heads=n_heads,
            depth=depth,
            max_seq_len=input_size,
            input_depth=input_depth,
            output_depth=datatreatment.get_output_length(),
            func_out='ReLU',
            learnable_pe=learnable_pe,
            no_pe=no_pe,
            residual=residual,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )
    else:
        model = CustomTransformer(
            hidden_size=hidden_size,
            dim_head=dim_head,
            n_heads=n_heads,
            depth=depth,
            max_seq_len=input_size,
            input_depth=input_depth,
            n_class=datatreatment.get_output_length(),
            softmax=softmax,
            learnable_pe=learnable_pe,
            network_type=model_type,
            no_pe=no_pe,
            residual=residual,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

    model.load_state_dict(model_parameters)

    model = model.float()
    model.to(device)

    model.eval()

    if task == 'map_rec':
        pass
    elif task == 'u_map' or task == 'bspline':
        entete = ["epoch", "id"]
        for l in range(input_size):
            entete += ["MASK" + str(l)]
            for ia, a in enumerate(datatreatment.get_output_length()):
                entete += [f"PRED{i}_{ia}_{l}" for i in range(a)]
                entete += [f"REEL{i}_{ia}_{l}" for i in range(a)]
        loggers_output.log(entete, False)
        total_iteration = len(val_loader)
        iteration = 1
        for item in val_loader:

            (input_vector, input_mask, input_len, output_vector,
             _, _) = datatreatment.data_treatment(item)

            with ((torch.set_grad_enabled(False))):
                outputs = model(src=input_vector, src_mask=input_mask, input_len=input_len)

                # print("res ", outputs[0])

                for i in range(outputs[0].size(0)):
                    log_line = [0, item['idx'][i].item()]

                    for k in range(outputs[0].size(1)):
                        log_line += [input_mask[i, k].int().item()]
                        for j, o in enumerate(outputs):
                            log_line += o[i, k, :].tolist() + output_vector[j][i, k, :].tolist()
                    loggers_output.log(log_line, False)
                loggers_output.save_log()
            print(f"Iteration {iteration}/{total_iteration}")
            iteration += 1

    else:
        for i in range(stack_u):
            entete = ["epoch", "id"]
            a = datatreatment.get_output_length()[i]
            entete += ["PRED" + str(i) for i in range(a)]
            entete += ["REEL" + str(i) for i in range(a)]
            entete += ["MASK" + str(i) for i in range(a)]

            loggers_output[i].log(entete, False)

        total_iteration = len(val_loader)
        iteration = 1

        for item in val_loader:

            input_vector, input_mask, input_len, output_vector, output_mask, output_mask_net = datatreatment.data_treatment(
                item)
            # print('input ', input_vector.shape)
            # print('output ', output_vector)
            # print('output mask ', output_mask)
            # print('output mask net', output_mask_net)

            with ((torch.set_grad_enabled(False))):
                outputs = model(src=input_vector, src_mask=input_mask, output_mask=output_mask_net,
                                input_len=input_len)
                # print("res ", outputs[0])

                for u in range(stack_u):

                    for i, _ in enumerate(outputs[0]):
                        log_line = [0, item['idx'][i].item()]

                        # for j, o in enumerate(outputs):
                        log_line += outputs[u][i].tolist() + output_vector[u][i].tolist() + output_mask[u][
                            i].int().tolist()
                        loggers_output[u].log(log_line, False)
                    loggers_output[u].save_log()
            print(f"Iteration {iteration}/{total_iteration}")
            iteration += 1

    del model
    torch.cuda.empty_cache()

    return 0
