import os
import torch
from fsspec.registry import default


def prep_param_network_args(default_object, args):
    if args.heads is not None and args.heads > 0:
        default_object['n_heads'] = args.heads

    if args.depth is not None and args.depth > 0:
        default_object['depth'] = args.depth

    if args.dim_head is not None and args.dim_head > 0:
        default_object['dim_head'] = args.dim_head

    if args.hidden_size is not None and args.hidden_size > 0:
        default_object['hidden_size'] = args.hidden_size

    if args.model_type is not None:
        default_object['model_type'] = args.model_type

    if args.learnable_pe is not None:
        default_object['learnable_pe'] = args.learnable_pe

    if args.no_pe is not None:
        default_object['no_pe'] = args.no_pe

    if args.no_residual is not None:
        default_object['residual'] = False

    if args.attn_dropout is not None:
        default_object['attn_dropout'] = args.attn_dropout

    if args.ff_dropout is not None:
        default_object['ff_dropout'] = args.ff_dropout

    return default_object


def parser_add_network_args(parser):
    parser.add_argument('--heads', type=int, help='Number of Self-attention heads of the network')
    parser.add_argument('--depth', type=int, help='Depth of the network')
    parser.add_argument('--dim_head', type=int, help='Depth dimension of the Self-attention heads')
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--model_type', type=str, choices={'classic', 'regression', 'clstoken'})
    parser.add_argument('--learnable_pe', action='store_true', default=None, help='Use learnable positional encoding')
    parser.add_argument('--no_pe', action='store_true', default=None, help='Do not use any positional encoding')
    parser.add_argument('--no-residual', action='store_true', default=None,
                        help='Do not use residual connection in self-attention')
    parser.add_argument('--attn_dropout', type=float, help='Dropout probability of self-attention heads')
    parser.add_argument('--ff_dropout', type=float, help='Dropout probability of feed-forward layer')

    return parser


def prep_param_utils_args(default_object, args):
    if args.workers_train is not None and args.workers_train >= 0:
        default_object['workers_train'] = args.workers_train

    if args.workers_val is not None and args.workers_val >= 0:
        default_object['workers_val'] = args.workers_val

    if args.lastsnap is not None:
        default_object['lastsnap'] = args.lastsnap

    if args.backbone_snapshot is not None:
        default_object['backbone_snapshot'] = args.backbone_snapshot

    if args.freeze_backbone is not None:
        default_object['freeze_backbone'] = args.freeze_backbone

    if args.fastmode is not None:
        default_object['fast_test'] = args.fastmode

    if args.nonverbose is not None:
        default_object['verbose_log'] = not args.nonverbose

    if args.snapfreq is not None and args.snapfreq >= 0:
        default_object['snapshot_frequency'] = args.snapfreq

    if args.notriton is not None and args.notriton:
        from bspline_curve_approximation_transformer.utils import triton_util
        triton_util.TRITON_AVAILABLE = False

    return default_object


def parser_add_utils_args(parser):
    parser.add_argument('--workers_train', type=int, default=0, help='Number of workers used during training')
    parser.add_argument('--workers_val', type=int, default=0, help='Number of workers used during validation')
    parser.add_argument('--snap', type=str, required=False, help='Path to snapshot file')
    parser.add_argument('--backbone_snapshot', type=str, help='Path to snapshot backbone file')
    parser.add_argument('--snapfreq', type=int, help='Snapshot output frequency')
    parser.add_argument('--output_folder', type=str, default='output', help='Path to output folder')
    parser.add_argument('--name', type=str, help='Force name of result folder')
    parser.add_argument('--surname', type=str, help='Surname of result folder')
    parser.add_argument('--lastsnap', type=str, help='File to record name of last snapshot during training')
    parser.add_argument('--no-cache-scheduler', action='store_true', help='Do not use cached scheduler')
    parser.add_argument('--no-cache-model', action='store_true', help='Do not use cached model')
    parser.add_argument('-f', '--fast', action='store_true', dest='fastmode', help="Use fast mode, only for testing")
    parser.add_argument('--notriton', action='store_true', help='Disable triton')
    parser.add_argument('--nonverbose', action='store_true', help='Disable console output for faster execution')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone during training')
    return parser


def prep_param_data_args(default_object, args):
    if args.task is not None:
        default_object['task'] = args.task

    if args.target_length is not None and args.target_length > 0:
        default_object['target_length'] = args.target_length

    if args.relative_coord is not None:
        default_object['relative_coord'] = args.relative_coord

    if args.keep_absolute_coord is not None:
        default_object['keep_absolute_coord'] = args.keep_absolute_coord

    if args.relative_coord_pos is not None:
        default_object['relative_coord_pos'] = args.relative_coord_pos

    if args.relative_output is not None:
        default_object['relative_output'] = args.relative_output

    if args.real_relative_output is not None:
        default_object['real_relative_output'] = args.real_relative_output

    if args.tl_len_factor is not None and args.tl_len_factor > 0:
        default_object['tl_len_factor'] = args.tl_len_factor

    if args.shifting is not None:
        default_object['shifting'] = args.shifting

    if args.shared_token is not None:
        default_object['shared_token'] = args.shared_token

    if args.stack_u is not None:
        default_object['stack_u'] = args.stack_u

    if args.force_dim is not None:
        default_object['force_dim'] = args.force_dim

    if args.add_time is not None:
        default_object['add_time'] = args.add_time

    if args.norm is not None:
        default_object['norm'] = args.norm

    if args.norm_unit is not None:
        default_object['norm_unit'] = args.norm_unit

    if args.noise_lvl is not None and args.noise_lvl >= 0.:
        default_object['noise_lvl'] = args.noise_lvl

    if args.noise_ratio is not None and args.noise_ratio >= 0.:
        default_object['noise_ratio'] = args.noise_ratio

    if args.normal_noise is not None and len(args.normal_noise) == 2:
        default_object['normal_noise'] = args.normal_noise

    if args.dataset_root_dir is not None:
        default_object['dataset_root_dir'] = args.dataset_root_dir

    if args.datasetminnodes is not None and args.datasetminnodes >= 0:
        default_object['dataset_min_nodes'] = args.datasetminnodes

    if args.softmax is not None:
        default_object['softmax'] = args.softmax

    return default_object


def parser_add_data_args(parser):
    parser.add_argument('--task', type=str, help='Task to train on', choices=['default', 'u_map'])
    parser.add_argument('--target_length', type=int, help='Input sampling length during training')
    parser.add_argument('--tl_len_factor', type=int, help='Scale factor of target length')
    parser.add_argument('--keep_absolute_coord', action='store_true', default=None,
                        help='Keep absolute coordinates on top of relative coordinates')
    parser.add_argument('--relative_coord', action='store_true', default=None,
                        help='Use relative coordinates for input')
    parser.add_argument('--relative_coord_pos', action='store_true', default=None,
                        help='Use relative coordinates positioning for input')
    parser.add_argument('--relative_output', action='store_true', default=None,
                        help='Use relative knots for output in u_map')
    parser.add_argument('--real_relative_output', action='store_true', default=None,
                        help='Use real relative knots for output in u_map')
    parser.add_argument('--shifting', action='store_true', default=None,
                        help='Randomly modifies curves during training')
    parser.add_argument('--shared_token', action='store_true', default=None)
    parser.add_argument('--stack_u', type=int)
    parser.add_argument('--force_dim', type=int)
    parser.add_argument('--add_time', action='store_true', default=None)
    parser.add_argument('--norm', action='store_true', dest='norm', default=None)
    parser.add_argument('--norm_unit', action='store_true', default=None)
    parser.add_argument('--noise_ratio', type=float)
    parser.add_argument('--noise_lvl', type=float)
    parser.add_argument('--normal_noise', nargs='*', type=float, default=None)
    parser.add_argument('--softmax', action='store_true', default=None, help='Use softmax on output')

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

    if args.scale_loss is not None:
        default_object['scale_loss'] = args.scale_loss

    if args.seed is not None and args.seed > 0:
        default_object['random_seed'] = args.seed

    if args.loss_scaling is not None:
        default_object['loss_scaling'] = args.loss_scaling

    if args.shuffle_seed is not None and args.shuffle_seed > 0:
        default_object['shuffle_seed'] = args.shuffle_seed

    if args.undo_arg is not None:
        default_object['undo_arg'] = args.undo_arg

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
    parser.add_argument('--scale_loss', action='store_true', default=None,
                        help='Use loss scaling for the multiple heads')
    parser.add_argument('--loss_scaling', type=int, default=None, nargs='+')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--shuffle_seed', type=int, help='Random seed for shuffling the dataset during training')

    parser.add_argument('--undo_arg', type=str, default=None, nargs='+')
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

    if args.target_length is not None and args.target_length >= 0:
        default_object['target_length'] = args.target_length

    if args.dataset_root_dir is not None:
        default_object['dataset_root_dir'] = args.dataset_root_dir

    if args.custom_output_folder is not None:
        default_object['custom_output_folder'] = args.custom_output_folder

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
    parser.add_argument('--dataset_root_dir', type=str)
    parser.add_argument('--custom_output_folder', type=str)

    return parser


class CustomInferenceArgs:
    def __init__(self, args: dict):
        assert args["snap"]
        self.snap: str = args.get("snap")

        self.workers_val = args.get("workers_val", None)
        self.bs = args.get("bs", None)
        self.k = args.get("k", 4)
        self.internal_knots = args.get("internal_knots", 1)
        self.n_segment = args.get("n_segment", 1)
        self.target_length = args.get("target_length", None)
        self.output_folder = args.get("output_folder", "output")

def prep_param_custom_inference_args(default_object, args):
    if args.workers_val is not None and args.workers_val >= 0:
        default_object['workers_val'] = args.workers_val

    if args.bs is not None and args.bs > 0:
        default_object['batch_size'] = args.bs

    if args.k is not None and args.k > 0:
        default_object['k'] = args.k

    if args.internal_knots is not None and args.internal_knots > 0:
        default_object['m'] = args.internal_knots

    if args.n_segment is not None and args.n_segment > 0:
        default_object['n_segment'] = args.n_segment

    if args.target_length is not None and args.target_length >= 0:
        default_object['target_length'] = args.target_length

    if args.output_folder is not None:
        default_object['output_folder'] = args.output_folder

    return default_object

def parser_add_custom_inference_args(parser):
    parser.add_argument('--workers_val', type=int, default=0, help='Number of workers used during validation')
    parser.add_argument('--snap', type=str, required=True, help='Path to snapshot file')

    parser.add_argument('--bs', type=int, help='Batch size')
    parser.add_argument('--k', type=int, help='Order of Bspline', default=4)
    parser.add_argument('--internal_knots', type=int, help='Number of internal knots', default=1)
    parser.add_argument('--n_segment', type=int, help='Number of segment per curve', default=1)
    parser.add_argument('--target_length', type=int, help='Forces input length if not 0')
    parser.add_argument('--output_folder', type=str, default='output')

    return parser


def prep_param_parnet_eval_args(default_object, args):
    default_object['output_folder'] = os.path.dirname(os.path.dirname(os.path.dirname(args.snap)))

    snapshot_object = torch.load(args.snap, map_location=default_object['device'])
    default_object.update(snapshot_object)

    default_object['training_name'] = os.path.basename(os.path.dirname(os.path.dirname(args.snap)))

    default_object['k'] = args.k
    default_object['m'] = args.m
    default_object['batch_size'] = args.bs
    default_object['output_length'] = args.output_length
    default_object['set_name'] = args.setname
    default_object['n_segment'] = args.n_segment
    default_object['segment_part'] = args.segment_part
    default_object['workers_val'] = args.workers_val

    default_object['segment'] = args.segment if args.segment is not None else False

    return default_object


def parser_add_parnet_eval_args(parser):
    parser.add_argument('--workers_val', type=int, default=0, help='Number of workers used during validation')
    parser.add_argument('--snap', type=str, required=True, help='Path to snapshot file')
    parser.add_argument('--bs', type=int, help='Batch size')

    parser.add_argument('--k', type=int, default=4, help="Order of bspline")
    parser.add_argument('--m', type=int, default=1, help="Number of internal knots")

    parser.add_argument('--output_length', type=int, default=0, help="")
    parser.add_argument('--seg', action='store_true', dest='segment')
    parser.add_argument('--n_segment', type=int, default=1)
    parser.add_argument('--segment_part', type=int, default=1)
    parser.add_argument('--setname', type=str, default="eval1")
    return parser


def prep_param_inference_args(default_object, args):
    if args.nonverbose is not None:
        default_object['verbose_log'] = not args.nonverbose

    default_object['k'] = args.k

    default_object['m'] = args.m

    return default_object


def parser_add_inference_args(parser):
    parser.add_argument('--nonverbose', action='store_true', help='Disable console output for faster execution')
    parser.add_argument('--snap', type=str, required=True, help='Path to snapshot file')

    parser.add_argument('--k', type=int, default=4, help="Order of bspline")
    parser.add_argument('--m', type=int, default=1, help="Number of internal knots")

    return parser
