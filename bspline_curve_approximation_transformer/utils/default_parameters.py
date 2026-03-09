from bspline_curve_approximation_transformer.version import version


def get_default_parameters() -> dict:
    return {
        'n_heads': 4,
        'depth': 4,
        'dim_head': 64,
        'hidden_size': 512,
        'model_type': 'classic',
        'learnable_pe': False,
        'no_pe': False,
        'residual': True,
        'ff_dropout': 0.,
        'attn_dropout': 0.,

        'training_name': 'BsplineDefaultParameters',
        'load_snapshot': False,
        'freeze_backbone': False,
        'backbone_snapshot': None,
        'model_parameters': None,
        'optimizer_cache': None,
        'scheduler_cache': None,

        'task': 'default',
        'target_length': 128,
        'tl_len_factor': 1,
        'force_dim': 2,
        'relative_coord': False,
        'relative_coord_pos': False,
        'keep_absolute_coord': False,
        'relative_output': False,
        'real_relative_output': False,
        'shifting': False,
        'shared_token': False,
        'stack_u': 0,
        'add_time': False,
        'norm': False,
        'norm_unit': False,
        'noise_ratio': 0.,
        'noise_lvl': 0.,
        'normal_noise': None,
        'snapshot_frequency': 5,

        'custom_output_folder': None,
        'dataset_root_dir': 'bspline_dataset',
        'dataset_min_nodes': 0,

        'starting_epoch': 0,
        'total_epochs': 0,
        'epochs': 10,
        'batch_size': 8,
        'lr': 0.05,
        'step': 5,
        'adam': False,
        'momentum': 0.9,
        'gamma': 0.75,
        'softmax': False,
        'scale_loss': False,
        'loss_scaling': None,
        'warmup_epochs': -1,

        'random_seed': 7364265666,
        'shuffle_seed': 2147483647,

    }


def task_specific_defaults(default_object: dict):
    if 'undo_arg' in default_object:
        for elem in default_object['undo_arg']:
            default_object[elem] = False
        default_object.pop('undo_arg')

    if 'custom_output_folder' in default_object:
        default_object.pop('custom_output_folder')

    if default_object['task'] == 'u_map':
        default_object['relative_coord'] = False
        default_object['softmax'] = False
        default_object['stack_u'] = 0
        default_object['shared_token'] = True
        default_object['scale_loss'] = False
    else:
        default_object['loss_scaling'] = None
        default_object['relative_output'] = False
        default_object['real_relative_output'] = False

    if default_object['keep_absolute_coord']:
        default_object['relative_coord_pos'] = True

    if default_object['relative_coord'] and default_object['relative_coord_pos']:
        default_object['relative_coord'] = False

    if 'eval_only' in default_object and default_object['eval_only']:
        default_object['epochs'] = 1
        default_object['starting_epoch'] = 0
        default_object['step'] = 1
        default_object['tl_len_factor'] = 1
        default_object['noise_ratio'] = 0.
        default_object['noise_lvl'] = 0.
        default_object['ff_dropout'] = 0.
        default_object['attn_dropout'] = 0.

    return default_object


def get_training_name(default_object: dict, args) -> str:
    if args.name is not None:
        return args.name

    return f"BSplineV{version}{args.surname if args.surname is not None else ''}" \
        + (f"_{default_object['task']}" if default_object['task'] != 'default' else '') \
        + f"_{default_object['target_length']}TL" \
        + (f"x{default_object['tl_len_factor']}" if default_object['tl_len_factor'] > 1 else '') \
        + ("_RC" if default_object['relative_coord'] else '') \
        + ("_RCP" if default_object['relative_coord_pos'] else '') \
        + ("_KAC" if default_object['keep_absolute_coord'] else '') \
        + ("_RO" if default_object['relative_output'] else '') \
        + ("_RRO" if default_object['real_relative_output'] else '') \
        + ("_S" if default_object['shifting'] else '') \
        + ("_ST" if default_object['shared_token'] else '') \
        + ("_T" if default_object['add_time'] else '') \
        + ("_N" if default_object['norm'] else '') \
        + ("_NU" if default_object['norm_unit'] else '') \
        + (f"_U{default_object['stack_u']}" if default_object['stack_u'] > 0 else '') \
        + (f"_{default_object['noise_ratio']}NZR" if default_object['noise_ratio'] > 0. and default_object[
            'noise_lvl'] > 0. else '') \
        + (f"x{default_object['noise_lvl']}" if default_object['noise_ratio'] > 0. and default_object[
            'noise_lvl'] > 0. else '') \
        + (f"NN{default_object['normal_noise'][0]}" +
           f"_{default_object['normal_noise'][1]}" if default_object['noise_ratio'] > 0. and
                                                      default_object['noise_lvl'] > 0. and
                                                      default_object['normal_noise'] is not None else '') \
        + ("_SL" if default_object['scale_loss'] or default_object['loss_scaling'] is not None else '') \
        + ("_SMX" if default_object['softmax'] else '') \
        + ("_ADM" if default_object['adam'] else '') \
        + ("_REG" if default_object['model_type'] == 'regression' else '') \
        + ("_CLS" if default_object['model_type'] == 'clstoken' else '') \
        + ("_LPE" if default_object['learnable_pe'] else '') \
        + ("_NPE" if default_object['no_pe'] else '') \
        + ("_NRC" if not default_object['residual'] else '') \
        + (f"_{default_object['attn_dropout']}ADP" if default_object['attn_dropout'] > 0. else '') \
        + (f"_{default_object['ff_dropout']}FDP" if default_object['ff_dropout'] > 0. else '') \
        + f"_{default_object['n_heads']}x{default_object['depth']}" \
        + f"_{default_object['dim_head']}DH" \
          f"_{default_object['hidden_size']}HSZ" \
          f"_{default_object['batch_size']}_{default_object['epochs']}" \
          f"_{default_object['step']}_{default_object['lr']}" \
          f"_{default_object['gamma']}_{default_object['momentum']}" \
        + (f"_{default_object['warmup_epochs']}WU"
           if default_object['warmup_epochs'] > 0 else "")
