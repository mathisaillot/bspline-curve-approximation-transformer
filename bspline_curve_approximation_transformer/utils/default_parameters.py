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

        'training_name': 'BsplineDefaultParameters',
        'load_snapshot': False,
        'model_parameters': None,
        'optimizer_cache': None,
        'scheduler_cache': None,

        'target_length': 128,
        'relative_coord': False,
        'shifting': False,
        'shared_token': False,
        'stack_u': 0,
        'add_time': False,
        'norm': False,
        'snapshot_frequency': 5,

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
        'warmup_epochs': -1,

        'random_seed': 7364265666,
        'shuffle_seed': 2147483647,

    }


def get_training_name(default_object: dict, args) -> str:
    if args.name is not None:
        return args.name

    return f"BSplineV{version}{args.surname if args.surname is not None else ''}" \
           f"_{default_object['target_length']}TL" \
        + ("_RC" if default_object['relative_coord'] else '') \
        + ("_S" if default_object['shifting'] else '') \
        + ("_ST" if default_object['shared_token'] else '') \
        + ("_T" if default_object['add_time'] else '') \
        + ("_N" if default_object['norm'] else '') \
        + (f"_U{default_object['stack_u']}" if default_object['stack_u'] > 0 else '') \
        + ("_SL" if default_object['scale_loss'] else '') \
        + ("_SMX" if default_object['softmax'] else '') \
        + ("_ADM" if default_object['adam'] else '') \
        + ("_REG" if default_object['model_type'] == 'regression' else '') \
        + ("_CLS" if default_object['model_type'] == 'clstoken' else '') \
        + ("_LPE" if default_object['learnable_pe'] else '') \
        + ("_NPE" if default_object['no_pe'] else '') \
        + f"_{default_object['n_heads']}x{default_object['depth']}" \
        + f"_{default_object['dim_head']}DH" \
          f"_{default_object['hidden_size']}HSZ" \
          f"_{default_object['batch_size']}_{default_object['epochs']}" \
          f"_{default_object['step']}_{default_object['lr']}" \
          f"_{default_object['gamma']}_{default_object['momentum']}" \
        + (f"_{default_object['warmup_epochs']}WU"
           if default_object['warmup_epochs'] > 0 else "")
