def prep_param_network_args(default_object, args):
    if args.heads is not None and args.heads > 0:
        default_object['n_heads'] = args.heads

    if args.depth is not None and args.depth > 0:
        default_object['depth'] = args.depth

    if args.dim_head is not None and args.dim_head > 0:
        default_object['dim_head'] = args.dimhead

    if args.hidden_size is not None and args.hidden_size > 0:
        default_object['hidden_size'] = args.hidsize

    if args.model_type is not None:
        default_object['model_type'] = args.model_type

    if args.learnable_pe is not None:
        default_object['learnable_pe'] = args.learnable_pe

    if args.no_pe is not None:
        default_object['no_pe'] = args.no_pe

    return default_object


def parser_add_network_args(parser):
    parser.add_argument('--heads', type=int, help='Number of Self-attention heads of the network')
    parser.add_argument('--depth', type=int, help='Depth of the network')
    parser.add_argument('--dim_head', type=int, help='Depth dimension of the Self-attention heads')
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--model_type', type=str, choices={'classic', 'regression', 'clstoken'})
    parser.add_argument('--learnable_pe', action='store_true', default=None, help='Use learnable positional encoding')
    parser.add_argument('--no_pe', action='store_true', default=None, help='Do not use any positional encoding')
    return parser


def prep_param_utils_args(default_object, args):
    if args.workers_train is not None and args.workers_train >= 0:
        default_object['workers_train'] = args.workers_train

    if args.workers_val is not None and args.workers_val >= 0:
        default_object['workers_val'] = args.workers_val

    if args.lastsnap is not None:
        default_object['lastsnap'] = args.lastsnap

    if args.fastmode is not None:
        default_object['fast_test'] = args.fastmode

    if args.nonverbose is not None:
        default_object['verbose_log'] = not args.nonverbose

    if args.snapfreq is not None and args.snapfreq >= 0:
        default_object['snapshot_frequency'] = args.snapfreq

    return default_object


def parser_add_utils_args(parser):
    parser.add_argument('--workers_train', type=int, default=0, help='Number of workers used during training')
    parser.add_argument('--workers_val', type=int, default=0, help='Number of workers used during validation')
    parser.add_argument('--snap', type=str, required=False, help='Path to snapshot file')
    parser.add_argument('--snapfreq', type=int, help='Snapshot output frequency')
    parser.add_argument('--output_folder', type=str, default='output', help='Path to output folder')
    parser.add_argument('--name', type=str, help='Force name of result folder')
    parser.add_argument('--surname', type=str, help='Surname of result folder')
    parser.add_argument('--lastsnap', type=str, help='File to record name of last snapshot during training')
    parser.add_argument('--no-cache-scheduler', action='store_true', help='Do not use cached scheduler')
    parser.add_argument('--no-cache-model', action='store_true', help='Do not use cached model')
    parser.add_argument('-f', '--fast', action='store_true', dest='fastmode', help="Use fast mode, only for testing")
    parser.add_argument('--nonverbose', action='store_true', help='Disable console output for faster execution')
    return parser


def prep_param_data_args(default_object, args):
    if args.target_length is not None and args.target_length > 0:
        default_object['target_length'] = args.target_length

    if args.relative_coord is not None:
        default_object['relative_coord'] = args.relative_coord

    if args.shifting is not None:
        default_object['shifting'] = args.shifting

    if args.shared_token is not None:
        default_object['shared_token'] = args.shared_token

    if args.stack_u is not None:
        default_object['stack_u'] = args.stack_u

    if args.add_time is not None:
        default_object['add_time'] = args.add_time

    if args.norm is not None:
        default_object['norm'] = args.norm

    if args.dataset_root_dir is not None:
        default_object['dataset_root_dir'] = args.dataset_root_dir

    if args.datasetminnodes is not None and args.datasetminnodes >= 0:
        default_object['dataset_min_nodes'] = args.datasetminnodes

    return default_object


def parser_add_data_args(parser):
    parser.add_argument('--target_length', type=int)
    parser.add_argument('--relative_coord', action='store_true', default=None)
    parser.add_argument('--shifting', action='store_true', default=None)
    parser.add_argument('--shared_token', action='store_true', default=None)
    parser.add_argument('--stack_u', type=int)
    parser.add_argument('--add_time', action='store_true', default=None)
    parser.add_argument('--norm', action='store_true', dest='norm', default=None)

    parser.add_argument('--datasetminnodes', type=int)
    parser.add_argument('--dataset_root_dir', type=str, default="bspline_dataset")
    return parser


def prep_param_training_args(default_object, args):
    if args.epochs is not None and args.epochs > 0:
        default_object['epochs'] = args.epochs

    if args.bs is not None and args.bs > 0:
        default_object['batch_size'] = args.bs

    if args.lr is not None and args.lr > 0:
        default_object['lr'] = args.lr

    if args.step is not None and args.step >= 0:
        default_object['step'] = args.step

    if args.warmup_epochs is not None:
        default_object['warmup_epochs'] = args.warmup_epochs

    if args.adam is not None:
        default_object['adam'] = args.adam

    if args.gamma is not None and args.gamma > 0:
        default_object['gamma'] = args.gamma

    if args.momentum is not None and args.momentum > 0:
        default_object['momentum'] = args.momentum

    if args.softmax is not None:
        default_object['softmax'] = args.softmax

    if args.scale_loss is not None:
        default_object['scale_loss'] = args.scale_loss

    if args.seed is not None and args.seed > 0:
        default_object['random_seed'] = args.seed

    if args.shuffle_seed is not None and args.shuffle_seed > 0:
        default_object['shuffle_seed'] = args.shuffle_seed

    return default_object


def parser_add_training_args(parser):
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--bs', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Starting learning rate')
    parser.add_argument('--step', type=int, help='Number of steps before restart of scheduler')
    parser.add_argument('--warmup_epochs', type=int, help='Number of warmup epochs')
    parser.add_argument('--adam', action='store_true', default=None, help='Use AdamW optimizer')
    parser.add_argument('--gamma', type=float, help='Learning rate decay factor')
    parser.add_argument('--momentum', type=float, help='SGD momentum or AdamW weight_decay')
    parser.add_argument('--softmax', action='store_true', default=None, help='Use softmax on output')
    parser.add_argument('--scale_loss', action='store_true', default=None,
                        help='Use loss scaling for the multiple heads')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--shuffle_seed', type=int, help='Random seed for shuffling the dataset during training')
    return parser


def prep_param_eval_args(default_object, args):
    if args.workers_val is not None and args.workers_val >= 0:
        default_object['workers_val'] = args.workers_val

    if args.fastmode is not None:
        default_object['fast_test'] = args.fastmode

    if args.nonverbose is not None:
        default_object['verbose_log'] = not args.nonverbose

    if args.bs is not None and args.bs > 0:
        default_object['batch_size'] = args.bs

    if args.target_length is not None and args.target_length > 0:
        default_object['target_length'] = args.target_length

    if args.dataset_root_dir is not None:
        default_object['dataset_root_dir'] = args.dataset_root_dir

    if args.datasetminnodes is not None and args.datasetminnodes >= 0:
        default_object['dataset_min_nodes'] = args.datasetminnodes

    return default_object


def parser_add_eval_args(parser):
    parser.add_argument('--workers_val', type=int, default=0, help='Number of workers used during validation')
    parser.add_argument('-f', '--fast', action='store_true', dest='fastmode', help="Use fast mode, only for testing")
    parser.add_argument('--nonverbose', action='store_true', help='Disable console output for faster execution')
    parser.add_argument('--snap', type=str, required=True, help='Path to snapshot file')

    parser.add_argument('--bs', type=int, help='Batch size')
    parser.add_argument('--target_length', type=int)
    parser.add_argument('--datasetminnodes', type=int)
    parser.add_argument('--dataset_root_dir', type=str, default="bspline_dataset")
    return parser
