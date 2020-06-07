from ast import literal_eval
import copy
import yaml
import numpy as np
from snmn.util.attr_dict import AttrDict

__C = AttrDict()
cfg = __C

# --------------------------------------------------------------------------- #
# general options
# --------------------------------------------------------------------------- #
# __C.EXP_NAME = '__default__'
# __C.GPU_ID = 0
# __C.GPU_MEM_GROWTH = True

# __C.VOCAB_QUESTION_FILE = './exp_clevr_snmn/data/vocabulary_clevr.txt'
# __C.VOCAB_LAYOUT_FILE = './exp_clevr_snmn/data/vocabulary_layout.txt'
# __C.VOCAB_ANSWER_FILE = './exp_clevr_snmn/data/answers_clevr.txt'
# __C.IMDB_FILE = './exp_clevr_snmn/data/imdb/imdb_%s.npy'

# __C.USE_FIXED_WORD_EMBED = False
# __C.FIXED_WORD_EMBED_FILE = ''

# --------------------------------------------------------------------------- #
# model options
# --------------------------------------------------------------------------- #
__C.MODEL = AttrDict()
__C.MODEL.H_FEAT = 14
__C.MODEL.W_FEAT = 14
__C.MODEL.T_CTRL = 7

__C.MODEL.BUILD_VQA = True
__C.MODEL.BUILD_LOC = False
__C.MODEL.H_IMG = 320  # size in loc
__C.MODEL.W_IMG = 480  # size in loc

# __C.MODEL.FEAT_DIM = 1024
__C.MODEL.EMBED_DIM = 400
__C.MODEL.LSTM_DIM = 160
# __C.MODEL.PE_DIM = 128
__C.MODEL.KB_DIM = 160
__C.MODEL.T_ENCODER = 45

__C.MODEL.CTRL = AttrDict()
__C.MODEL.CTRL.LINEAR_MODULE_WEIGHTS = False
__C.MODEL.CTRL.MLP_MODULE_WEIGHTS_BIAS_TERM = True
__C.MODEL.CTRL.NORMALIZE_ATT = True
__C.MODEL.CTRL.USE_GUMBEL_SOFTMAX = False
__C.MODEL.CTRL.GUMBEL_SOFTMAX_TMP = 0.5
__C.MODEL.CTRL.USE_WORD_EMBED = False
__C.MODEL.CTRL.USE_HARD_ARGMAX_LAYOUT = False

__C.MODEL.NMN = AttrDict()
__C.MODEL.NMN.MEM_DIM = 256
__C.MODEL.NMN.STACK = AttrDict()
__C.MODEL.NMN.STACK.LENGTH = 4
__C.MODEL.NMN.STACK.USE_HARD_SHARPEN = True
__C.MODEL.NMN.STACK.SOFT_SHARPEN_TEMP = 0.2
__C.MODEL.NMN.STACK.GUARD_STACK_PTR = True
__C.MODEL.NMN.VALIDATE_MODULES = False
__C.MODEL.NMN.HARD_MODULE_VALIDATION = False
__C.MODEL.NMN.DESCRIBE_ONE = AttrDict()
__C.MODEL.NMN.DESCRIBE_ONE.KEEP_STACK = False
__C.MODEL.NMN.DESCRIBE_TWO = AttrDict()
__C.MODEL.NMN.DESCRIBE_TWO.KEEP_STACK = False

# __C.MODEL.VQA_OUTPUT_DIM = 512
# __C.MODEL.VQA_OUTPUT_USE_QUESTION = True
# __C.MODEL.BBOX_REG_USE_QUESTION = False
# __C.MODEL.LOC_SCORES_POS_AFFINE = False
# __C.MODEL.BBOX_REG_AS_FCN = False

__C.MODEL.REC = AttrDict()
__C.MODEL.REC.USE_REC_LOSS = False
__C.MODEL.REC.USE_LOGITS = True
__C.MODEL.REC.USE_TXT_ATT = True

# --------------------------------------------------------------------------- #
# training options
# --------------------------------------------------------------------------- #
# __C.TRAIN = AttrDict()
# __C.TRAIN.SPLIT_VQA = 'train'
# __C.TRAIN.SPLIT_LOC = 'loc_train'
# __C.TRAIN.BATCH_SIZE = 64
# __C.TRAIN.USE_GT_LAYOUT = True
# __C.TRAIN.WEIGHT_DECAY = 1e-5
# __C.TRAIN.DROPOUT_KEEP_PROB = 0.85
# __C.TRAIN.SOLVER = AttrDict()
# __C.TRAIN.SOLVER.LR = 1e-4
# __C.TRAIN.EMV_DECAY = 0.999
# __C.TRAIN.START_ITER = 0

# __C.TRAIN.VQA_LOSS_WEIGHT = 1.
# __C.TRAIN.BBOX_IND_LOSS_WEIGHT = 1.
# __C.TRAIN.BBOX_OFFSET_LOSS_WEIGHT = 1.
# __C.TRAIN.LAYOUT_LOSS_WEIGHT = 1.
# __C.TRAIN.REC_LOSS_WEIGHT = 1.

# __C.TRAIN.SNAPSHOT_DIR = './exp_clevr_snmn/tfmodel/%s/'
# __C.TRAIN.SNAPSHOT_INTERVAL = 10000
# __C.TRAIN.INIT_FROM_WEIGHTS = False
# __C.TRAIN.INIT_WEIGHTS_FILE = ''
# __C.TRAIN.MAX_ITER = 400000
# __C.TRAIN.LOG_DIR = './exp_clevr_snmn/tb/%s/'
# __C.TRAIN.LOG_INTERVAL = 20

# __C.TRAIN.BBOX_IOU_THRESH = .5

# __C.TRAIN.USE_SHARPEN_LOSS = False
# __C.TRAIN.SHARPEN_LOSS_TYPE = 'entropy'
# __C.TRAIN.SHARPEN_LOSS_WEIGHT = 1e-5
# __C.TRAIN.SHARPEN_LOSS_SCALING_TYPE = 'warmup_scaling'
# __C.TRAIN.SHARPEN_LOSS_SCALING_FUNC = ''
# __C.TRAIN.SHARPEN_SCHEDULE_BEGIN = 500
# __C.TRAIN.SHARPEN_SCHEDULE_END = 10000

# # --------------------------------------------------------------------------- #
# # test options
# # --------------------------------------------------------------------------- #
# __C.TEST = AttrDict()
# __C.TEST.BATCH_SIZE = 64
# __C.TEST.USE_EMV = True
# __C.TEST.SPLIT_VQA = 'val'
# __C.TEST.SPLIT_LOC = 'loc_val'
# __C.TEST.SNAPSHOT_FILE = './exp_clevr_snmn/tfmodel/%s/%08d'
# __C.TEST.ITER = -1  # Needs to be supplied

# __C.TEST.RESULT_DIR = './exp_clevr_snmn/results/%s/%08d'
# __C.TEST.OUTPUT_VQA_EVAL_PRED = True
# __C.TEST.VIS_SEPARATE_CORRECTNESS = False
# __C.TEST.NUM_VIS = 100
# __C.TEST.NUM_VIS_CORRECT = 50
# __C.TEST.NUM_VIS_INCORRECT = 50
# __C.TEST.VIS_DIR_PREFIX = 'vis'
# __C.TEST.STEPWISE_VIS = True  # Use the (new) stepwise visualization
# __C.TEST.VIS_SHOW_ANSWER = True
# __C.TEST.VIS_SHOW_STACK = True
# __C.TEST.VIS_SHOW_IMG = True

# __C.TEST.BBOX_IOU_THRESH = .5

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
