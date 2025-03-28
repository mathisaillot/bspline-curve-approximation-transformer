import copy
import os
import random
import time
from datetime import datetime as dt
from math import ceil, inf

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from torch.optim import lr_scheduler, AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchmetrics import MeanSquaredLogError

from bspline_curve_approximation_transformer.data import BSplineSequenceDataset, factory_data_transforms, \
    BSplineSequenceDatasetWithTransform, DataPretreator
from bspline_curve_approximation_transformer.transformer import CustomTransformer
from bspline_curve_approximation_transformer.utils import Logger, LoggerCSV


def format_time_measure(time):
    fullminutes = time // 60
    minutes = fullminutes % 60
    hours = fullminutes // 60
    seconds = time % 60
    return f'{hours: .0f}h {minutes: 2.0f}m {seconds: 2.2f}s'


def train_model(model,
                train_loader: DataLoader,
                val_loader: DataLoader,
                data_pretreator: DataPretreator,
                dataset_sizes,
                device: torch.device,
                criterion,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                softmax: bool,
                num_epochs: int,
                logger_metrics: Logger,
                logger_info: Logger,
                logger_output: LoggerCSV,
                snapshot_object: dict,
                snapshot_folder: str,
                snapshot_frequency: int = 10,
                starting_epoch: int = 0,
                total_epochs: int = 0,
                verbose_log: bool = True,
                return_best: bool = False,
                scale_loss: bool = False,
                eval_only: bool = False,
                lastsnap: str = None):
    train = 'train'
    val = 'val'
    phases = [train, val]

    dataloadertv = {train: train_loader,
                    val: val_loader}
    if eval_only:
        phases = [val]

    time_measures = {'begin': time.time()}

    output_len = len(data_pretreator.get_output_length())

    single_output = output_len == 1

    if single_output:
        epch_log = ["loss_" + train, "Metric_" + train, "loss_" + val, "Metric_" + val]
    else:
        epch_log = ["loss_" + train] + [f'loss_{a + 1}_{train}' for a in range(output_len)] + \
                   ["loss_" + val] + [f'loss_{a + 1}_{val}' for a in range(output_len)]

    logger_metrics.log(epch_log, False)

    epch_log = []
    total_units = {}

    global_metrics = {}

    for i in phases:
        global_metrics[i] = MeanSquaredLogError().to(device)
        total_units[i] = (num_epochs - starting_epoch) * dataset_sizes[i]
        time_measures['units' + i] = 0
        time_measures[i + 'total'] = 0
        time_measures[i + 'frequency'] = 0

    if return_best:
        best_model_wts = copy.deepcopy(model.state_dict())
        best_metric = inf
        best_epoch = 0
        best_optimizer = None

    entete = ["epoch", "id"]
    for a in data_pretreator.get_output_length():
        entete += ["PRED" + str(i) for i in range(a)]
        entete += ["REEL" + str(i) for i in range(a)]
        entete += ["MASK" + str(i) for i in range(a)]

    logger_output.log(entete, False)

    time_measures['now'] = time.time()

    for epoch in range(starting_epoch, num_epochs):
        total_epochs = total_epochs + 1
        time_measures['epoch_start'] = time_measures['now']
        time_measures['last_dump'] = time_measures['now']

        logger_info.log('#' * 50 + f'   Epoch {epoch + 1}/{num_epochs}   ' + '#' * 50)

        # Each epoch has a training and validation phase
        for phase in phases:
            time_measures[phase + '_start'] = time_measures['now']
            if phase == train:
                model.train()  # Set model to training mode
                is_training = True
            else:
                model.eval()  # Set model to evaluate mode
                is_training = False

            total_iteration = len(dataloadertv[phase])
            iteration = 1
            running_loss = 0.0
            running_loss_count = 0
            running_losses = [0.0 for _ in range(output_len)]
            running_losses_count = [0 for _ in range(output_len)]
            output_loss = []
            loss = 0.0
            local_n = 0
            metric = 0.0
            # Iterate over data.
            for item in dataloadertv[phase]:
                time_measures['units' + phase] += dataloadertv[phase].batch_size
                time_measures['iteration'] = time_measures['now']

                (input_vector, input_mask, output_vector,
                 output_mask, output_mask_net) = data_pretreator.data_treatment(item)

                # zero the parameter gradients
                optimizer.zero_grad()

                with ((torch.set_grad_enabled(is_training))):
                    outputs = model(src=input_vector, src_mask=input_mask, output_mask=output_mask_net)

                    output_loss = [criterion(outputs[i], output_vector[i]) for i in range(output_len)]
                    for i in range(output_len):
                        if len(output_loss[i].shape) != 0:
                            output_loss[i] = (output_loss[i] * output_mask[i]).sum()
                            bs_count = torch.count_nonzero(torch.sum(output_mask[i], 1))
                            if bs_count > 0:
                                running_losses_count[i] += bs_count
                                if not scale_loss:
                                    output_loss[i] = output_loss[i] / output_mask[i].sum()
                                else:
                                    output_loss[i] = output_loss[i] / bs_count

                                running_losses[i] += output_loss[i].item() * bs_count
                        if i == 0:
                            loss = output_loss[i] * bs_count
                        else:
                            loss += output_loss[i] * bs_count

                        loss /= input_vector.size(0)
                        tot_loss = loss

                    if is_training:
                        tot_loss.backward()
                        optimizer.step()

                with torch.set_grad_enabled(False):
                    if single_output:
                        local_loss = criterion(outputs[0], output_vector[0])
                        local_mask = output_mask_net[0] if softmax else torch.logical_and(output_vector[0] > -0.5,
                                                                                          output_vector[0] < 0.5) * \
                                                                        output_mask[0]
                        sum_local_mask = local_mask.sum()
                        local_metric = (local_loss * local_mask).sum() / (sum_local_mask if sum_local_mask > 0 else 1)
                        local_n += sum_local_mask
                        metric += local_metric * sum_local_mask

                time_measures['now'] = time.time()
                time_measures['iteration'] = time_measures['now'] - time_measures['iteration']

                time_measures[phase + 'total'] += time_measures['iteration']
                time_measures[phase + 'frequency'] = time_measures[phase + 'total'] / time_measures['units' + phase]

                time_measures["estimate"] = 0
                if not eval_only:
                    time_measures["estimate"] = (total_units[train] - time_measures['units' + train]) \
                                                * time_measures[train + 'frequency'] + \
                                                (total_units[val] - time_measures['units' + val]) \
                                                * (time_measures[val + 'frequency'] if time_measures[
                                                                                           val + 'frequency'] > 0 else
                                                   time_measures[train + 'frequency'] / 2)

                loss_log = ''
                for i in range(output_len):
                    loss_log += f'| Loss_{i + 1} {output_loss[i].item():.8f} '
                logger_info.log(f'{dt.now().isoformat()} '
                                f'# Epoch {epoch + 1}/{num_epochs} '
                                f'# {phase} {iteration}/{total_iteration} '
                                f'| Loss {loss.item():.8f} '
                                + (f'| Metric {local_metric.item():.8f} ' if single_output else f'{loss_log}') +
                                (
                                    f'| LR {optimizer.param_groups[0]["lr"]:.20f} ' if is_training else '') +
                                f'| Time {time_measures["iteration"]:.3f}s '
                                f'| Temps restant {format_time_measure(time_measures["estimate"])}',
                                verbose_log)

                running_loss += loss.item() * input_vector.size(0)

                iteration += 1
                if time_measures['now'] - time_measures['last_dump'] > 300:
                    logger_info.save_log()
                    time_measures['last_dump'] = time_measures['now']

                if not is_training and ((snapshot_frequency > 0
                                         and (epoch % snapshot_frequency == (snapshot_frequency - 1)))
                                        or (epoch == num_epochs - 1)):
                    for i, _ in enumerate(outputs[0]):
                        log_line = [epoch, item['idx'][i].item()]

                        for j, o in enumerate(outputs):
                            log_line += o[i].tolist() + output_vector[j][i].tolist() + output_mask[j][i].int().tolist()
                        logger_output.log(log_line, False)
                    logger_output.save_log()

                if is_training:
                    scheduler.step()

            if single_output:
                metric = metric / (local_n if local_n > 0 else 1)

            epoch_loss = running_loss / dataset_sizes[phase]

            epch_log.append(epoch_loss)
            if single_output:
                epch_log.append(metric.item())
            else:
                epoch_losses = [
                    running_losses[i] / (running_losses_count[i] if running_losses_count[i] > 0 else 1) for
                    i in range(len(running_losses))]
                log_out = ''
                for i, e in enumerate(epoch_losses):
                    if torch.is_tensor(e):
                        e = e.item()
                    epch_log.append(e)
                    log_out += f'| Loss_{i + 1}: {e:.8f} '

            time_measures['now'] = time.time()
            time_measures[phase] = time_measures['now'] - time_measures[phase + '_start']

            logger_info.log(f'{dt.now().isoformat()} '
                            f'# Epoch {epoch + 1}/{num_epochs} '
                            f'# Fin {phase} '
                            f'| Loss: {epoch_loss:.8f} '
                            + (f'| Metric: {metric: .8f} ' if single_output else f'{log_out}') +
                            f'| Time {format_time_measure(time_measures[phase])} '
                            f'| Temps restant {format_time_measure(time_measures["estimate"])}')
            logger_info.log('-' * 150)
            logger_info.save_log()
            time_measures['last_dump'] = time_measures['now']

            # deep copy the model
            if return_best and not is_training and best_metric > epoch_loss:
                best_metric = epoch_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optimizer = best_optimizer

        logger_metrics.log(epch_log, False)
        logger_metrics.save_log()
        epch_log = []

        # snapshot
        if not eval_only and ((snapshot_frequency > 0 and (epoch % snapshot_frequency == (snapshot_frequency - 1)))
                              or (epoch == num_epochs - 1)):
            snapshot_object['starting_epoch'] = epoch + 1
            snapshot_object['total_epochs'] = total_epochs
            snapshot_object['model_parameters'] = copy.deepcopy(model.state_dict())
            snapshot_object['optimizer_cache'] = optimizer
            snapshot_object['scheduler_cache'] = None  # todo fix scheduler log

            snapshot_name = os.path.join(snapshot_folder, f'snap_{epoch + 1}.pt')
            logger_info.log('SAVING SNAPSHOT : ' + snapshot_name)
            torch.save(snapshot_object, snapshot_name)
            if lastsnap is not None:
                f = open(lastsnap, mode="w")
                if f:
                    f.write(snapshot_name)
                f.close()

    time_elapsed = time.time() - time_measures["begin"]

    logger_info.log(f'Training complete in {format_time_measure(time_elapsed)}')

    # load best model weights
    if return_best and not eval_only:
        model.load_state_dict(best_model_wts)
        if best_epoch != num_epochs:
            snapshot_object['starting_epoch'] = best_epoch
            snapshot_object['total_epochs'] = total_epochs - num_epochs + best_epoch
            snapshot_object['model_parameters'] = best_model_wts
            snapshot_object['optimizer_cache'] = best_optimizer
            snapshot_object['scheduler_cache'] = None  # todo fix scheduler log

            snapshot_name = os.path.join(snapshot_folder, f'snap_be_{best_epoch}.pt')
            logger_info.log('SAVING BEST SNAPSHOT : ' + snapshot_name)
            torch.save(snapshot_object, snapshot_name)

    logger_info.save_log()
    logger_metrics.save_log()
    logger_output.save_log()

    return model


def setup_dataflow(dataset_train, dataset_eval, train_idx, val_idx, batch_size, workers_train, workers_val,
                   shuffle_seed):
    g_cpu = torch.Generator()

    train_sampler = SubsetRandomSampler(train_idx, g_cpu.manual_seed(shuffle_seed))
    val_sampler = SubsetRandomSampler(val_idx, g_cpu.manual_seed(shuffle_seed))

    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, num_workers=workers_train)
    val_loader = DataLoader(dataset_eval, batch_size=batch_size, sampler=val_sampler, num_workers=workers_val)

    return train_loader, val_loader


def training_setup(device: torch.device,
                   dataset_root_dir: str,
                   target_length: int,
                   dataset_min_nodes: int,
                   norm: bool,
                   output_folder: str,
                   random_seed: int,
                   shuffle_seed: int,
                   relative_coord: bool,
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
                   add_time: bool,
                   scale_loss: bool,
                   stack_u: int,
                   lr: float,
                   momentum: float,
                   gamma: float,
                   step: int,
                   warmup_epochs: int,
                   snapshot_frequency: int,
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
                   lastsnap: str = None):
    torch.cuda.empty_cache()
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    # rng = np.random.default_rng(seed=random_seed)

    print(f"STARTING ON DEVICE : {device}")

    print(f"LOADING DATASET {dataset_root_dir}")

    dataset = BSplineSequenceDataset(target_length=target_length, root_dir=dataset_root_dir,
                                     min_nodes_length=dataset_min_nodes, mini_seed=random_seed, padding_u=stack_u == 0)

    transforms = factory_data_transforms(relative_coord=relative_coord, norm=norm, softmax=softmax,
                                         shifting=shifting, echant=True, shared_token=shared_token, add_time=add_time,
                                         stack_u=stack_u, target_length=target_length)

    # dataset
    dataset_train = BSplineSequenceDatasetWithTransform(dataset, transform=transforms['train'])
    dataset_eval = BSplineSequenceDatasetWithTransform(dataset, transform=transforms['val'])

    datatreatment = DataPretreator(device=device, dataset=dataset_train)

    train_idx = dataset.get_train_idx()
    val_idx = dataset.get_val_idx()

    if fast_test:
        np.random.shuffle(train_idx)
        train_idx = train_idx[:64]
        np.random.shuffle(val_idx)
        val_idx = val_idx[:8]

    if not eval_only:
        log_folder_1 = os.path.join(output_folder, training_name + "_" + dt.now().strftime('%Y-%m-%dT%H-%M-%S'))
        os.mkdir(log_folder_1)

    input_size = datatreatment.get_input_length()
    input_depth = datatreatment.get_input_depth()

    criterion = nn.MSELoss(reduction='none')

    snapshot_object = {
        'random_seed': random_seed,
        'shuffle_seed': shuffle_seed,
        'relative_coord': relative_coord,
        'norm': norm,
        'dataset_root_dir': dataset_root_dir,
        'dataset_min_nodes': dataset_min_nodes,
        'target_length': target_length,
        'epochs': epochs,
        'total_epochs': total_epochs,
        'batch_size': batch_size,
        'hidden_size': hidden_size,
        'n_heads': n_heads,
        'model_type': model_type,
        'softmax': softmax,
        'adam': adam,
        'shifting': shifting,
        'shared_token': shared_token,
        'learnable_pe': learnable_pe,
        'no_pe': no_pe,
        'stack_u': stack_u,
        'add_time': add_time,
        'scale_loss': scale_loss,
        'depth': depth,
        'dim_head': dim_head,
        'lr': lr,
        'momentum': momentum,
        'gamma': gamma,
        'step': step,
        'warmup_epochs': warmup_epochs,
        'snapshot_frequency': snapshot_frequency,
        'training_name': training_name,
        'load_snapshot': False,
        'model_parameters': None,
        'optimizer_cache': None,
        'scheduler_cache': None,
        'starting_epoch': 0,
        'snap': snap,
    }

    if not eval_only:
        torch.save(snapshot_object, os.path.join(log_folder_1, 'Parameters.pt'))

    snapshot_object['load_snapshot'] = True

    dataset_sizes = {"train": len(train_idx),
                     "val": len(val_idx)}

    train_loader, val_loader = setup_dataflow(
        dataset_train,
        dataset_eval,
        train_idx.flatten(),
        val_idx.flatten(),
        batch_size=batch_size,
        workers_train=workers_train,
        workers_val=workers_val,
        shuffle_seed=shuffle_seed,
    )

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
    )
    from_snap = load_snapshot and model_parameters is not None

    if from_snap:
        model.load_state_dict(model_parameters)

    model = model.float()
    model.to(device)

    load_optimizer = load_snapshot and optimizer_cache is not None

    if load_optimizer:
        optimizer_ft = optimizer_cache
    else:
        optimizer_ft = AdamW(params=model.parameters(), lr=lr, weight_decay=momentum) if adam \
            else optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    load_scheduler = load_snapshot and scheduler_cache is not None

    if load_scheduler:
        exp_lr_scheduler = scheduler_cache
    else:
        epoch_steps = len(train_loader)
        exp_lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer_ft, num_training_steps=epochs * epoch_steps,
            num_cycles=1 if epochs <= step else epochs // step,
            num_warmup_steps=warmup_epochs * epoch_steps if warmup_epochs >= 0 else (
                epoch_steps if total_epochs == 0 else 0)) if adam \
            else lr_scheduler.StepLR(optimizer_ft, step_size=step * epoch_steps, gamma=gamma)

    if not eval_only:
        logger_data = LoggerCSV(os.path.join(log_folder_1, 'metrics.csv'))
        logger_output = LoggerCSV(os.path.join(log_folder_1, 'output.csv'))
        logger_info = Logger(os.path.join(log_folder_1, 'info.txt'))
        log_folder_snapshot = os.path.join(log_folder_1, 'snapshots')
        os.mkdir(log_folder_snapshot)
    else:
        logger_data = LoggerCSV(os.path.join(output_folder, f'metrics_{target_length}.csv'))
        logger_output = LoggerCSV(os.path.join(output_folder, f'output_{target_length}.csv'))
        logger_info = Logger(os.path.join(output_folder, f'info_{target_length}.txt'))
        log_folder_snapshot = os.path.join(output_folder, 'snapshots')

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger_info.log(f"Starting training for {epochs} Epochs |"
                    f" batch size {batch_size} | "
                    f"lr {lr} | step {step} | gamma {gamma}"
                    + (f" | warmup {warmup_epochs}" if warmup_epochs >= 0 else "")
                    + f" | momentum {momentum}  " + ("LOAD OPTIMIZER | " if load_optimizer else "") +
                    ("LOAD SCHEDULER | " if load_scheduler else "") +
                    f"| seed {random_seed}\n  Dataset {dataset_root_dir} Total Epochs {total_epochs}"
                    f"| Train size {len(train_idx)}"
                    f" | Val size {len(val_idx)} " +
                    (" RelativeCoord " if relative_coord else "") +
                    (" Timed " if add_time else "") +
                    (" Normalized " if norm else "") +
                    (" Scaled Loss " if scale_loss else "") +
                    (" Softmax " if softmax else "") +
                    (f" Stacked U {stack_u} " if stack_u > 0 else "") +
                    (" Adam " if softmax else "") +
                    (" Shifting " if shifting else "") +
                    (" SharedToken " if shared_token else "") +
                    f" | Target Length {target_length}\n"
                    f" | Output size {datatreatment.get_output_length()}\n"
                    + (f"   FROM SNAPSHOT {snap} starting at {starting_epoch}\n" if from_snap else "")
                    + f"    Model {model_type if model_type != 'classic' else ''}"
                    + f"{' learnable positional encoding' if learnable_pe else ''}"
                    + f"{' no positional encoding' if no_pe else ''}"
                      f": hidden size {hidden_size} | heads {n_heads} "
                      f"| depth {depth} | dim_head {dim_head}\n"
                    + (f"   FAST\n" if fast_test else "")
                    + (f"   SILENT\n" if not verbose_log else ""))

    logger_info.log(f"Total estimated parameters : {pytorch_total_params}")

    logger_info.save_log()

    dataset.reset(total_epochs)

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        data_pretreator=datatreatment,
        dataset_sizes=dataset_sizes,
        device=device,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=exp_lr_scheduler,
        num_epochs=epochs,
        logger_metrics=logger_data,
        logger_info=logger_info,
        logger_output=logger_output,
        snapshot_object=snapshot_object,
        snapshot_folder=log_folder_snapshot,
        snapshot_frequency=snapshot_frequency,
        starting_epoch=starting_epoch,
        verbose_log=verbose_log,
        softmax=softmax,
        return_best=True,
        total_epochs=total_epochs,
        scale_loss=scale_loss,
        eval_only=eval_only,
        lastsnap=lastsnap
    )

    del model
    torch.cuda.empty_cache()
