import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T, newaxis as ax

from snmn.config import cfg
from snmn.util.cnn import fc_layer as fc, fc_elu_layer as fc_elu
from snmn.util.gumbel_softmax import gumbel_softmax


class Controller:

  def __init__( self, main_config, lstm_seq, q_encoding, embed_seq, seq_length_batch,
                num_module, is_train, scope='controller', reuse=None, gt_module_ids=None):
    """
    Build the controller that is used to give inputs to the neural modules.
    The controller unrolls itself for a fixed number of time steps.
    All parameters are shared across time steps.

    # The controller uses an auto-regressive structure like a GRU cell.
    # Attention is used over the input sequence.
    Here, the controller is the same as in the previous MAC paper, and
    additional module weights

    Input:
      lstm_seq: [N, S, d], tf.float32
      q_encoding: [N, d], tf.float32
      embed_seq: [N, S, e], tf.float32
      seq_length_batch: [N], tf.int32
    """
    self.is_train = is_train
    dim = cfg.MODEL.LSTM_DIM
    ctrl_dim = (cfg.MODEL.EMBED_DIM if cfg.MODEL.CTRL.USE_WORD_EMBED
          else cfg.MODEL.LSTM_DIM)
    T_ctrl = cfg.MODEL.T_CTRL
    
    # an attention mask to normalize textual attention over the actual
    # sequence length
    lstm_seq = tf.transpose(lstm_seq, perm=[1, 0, 2]) # Transpose to [S, N, d]
    embed_seq = tf.transpose(embed_seq, perm=[1, 0, 2])
    S = tf.shape(lstm_seq)[0]
    N = tf.shape(lstm_seq)[1]
    # att_mask: [S, N, 1]
    att_mask = tf.less(tf.range(S)[:, ax, ax], seq_length_batch[:, ax])
    att_mask = tf.cast(att_mask, tf.float32)
    with tf.variable_scope(scope, reuse=reuse):
      S = tf.shape(lstm_seq)[0]
      N = tf.shape(lstm_seq)[1]

      # manually unrolling for a number of timesteps
      c_init = tf.get_variable(
        'c_init', [1, ctrl_dim],
        initializer=tf.initializers.random_normal(
          stddev=np.sqrt(1. / ctrl_dim)))
      c_prev = tf.tile(c_init, to_T([N, 1]))
      c_prev.set_shape([None, ctrl_dim])
      c_list = []
      cv_list = []
      c_st_list = []
      module_logit_list = []
      module_prob_list = []
      for t in range(T_ctrl):
        q_i = fc('fc_q_%d' % t, q_encoding, output_dim=dim, is_train=self.is_train, dropout=main_config.nmn_dropout)  # [N, d]
        q_i_c = tf.concat([q_i, c_prev], axis=1)  # [N, 2d]
        cq_i = fc('fc_cq', q_i_c, output_dim=dim, is_train=self.is_train, reuse=(t > 0), dropout=main_config.nmn_dropout)
        c_st_list.append(cq_i)
        if cfg.MODEL.CTRL.LINEAR_MODULE_WEIGHTS:
          # Apply a linear transform on cq_i to predict the module
          # weights
          module_logit = fc(
            'fc_module_w_linear', cq_i, output_dim=num_module, is_train=self.is_train,
            bias_term=False, reuse=(t > 0), dropout=main_config.nmn_dropout)  # [N, M]
        else:
          # Apply a fully connected network on top of cq_i to predict
          # the module weights
          bias_term = cfg.MODEL.CTRL.MLP_MODULE_WEIGHTS_BIAS_TERM
          module_w_l1 = fc_elu(
            'fc_module_w_layer1', cq_i, output_dim=dim, is_train=self.is_train,
            reuse=(t > 0), dropout=main_config.nmn_dropout)
          module_logit = fc(
            'fc_module_w_layer2', module_w_l1,
            output_dim=num_module, is_train=self.is_train, bias_term=bias_term,
            reuse=(t > 0), dropout=main_config.nmn_dropout)  # [N, M]
        module_logit_list.append(module_logit)
        if cfg.MODEL.CTRL.USE_GUMBEL_SOFTMAX:
          module_prob = gumbel_softmax(
            module_logit, cfg.MODEL.CTRL.GUMBEL_SOFTMAX_TMP)
        else:
          module_prob = tf.nn.softmax(module_logit, axis=1)
        # Use hard (discrete) layout if specified
        if cfg.MODEL.CTRL.USE_HARD_ARGMAX_LAYOUT:
          if t == 0:
            print('Using hard argmax layout. '
                'This should only be used at test time!')
          module_prob = tf.one_hot(
            tf.argmax(module_prob, axis=-1), num_module,
            dtype=module_prob.dtype)
        module_prob_list.append(module_prob)

        elem_prod = tf.reshape(cq_i * lstm_seq, to_T([S*N, dim]))
        elem_prod.set_shape([None, dim])  # [S*N, d]
        raw_cv_i = tf.reshape(
          fc('fc_cv_i', elem_prod, output_dim=1, is_train=self.is_train, reuse=(t > 0), dropout=main_config.nmn_dropout),
          to_T([S, N, 1]))
        cv_i = tf.nn.softmax(raw_cv_i, axis=0)  # [S, N, 1]
        # normalize the attention over the actual sequence length
        if cfg.MODEL.CTRL.NORMALIZE_ATT:
          cv_i = cv_i * att_mask
          cv_i /= tf.reduce_sum(cv_i, 0, keepdims=True)

        if cfg.MODEL.CTRL.USE_WORD_EMBED:
          c_i = tf.reduce_sum(cv_i * embed_seq, axis=[0])  # [N, e]
        else:
          c_i = tf.reduce_sum(cv_i * lstm_seq, axis=[0])  # [N, d]
        c_list.append(c_i)
        cv_list.append(cv_i)
        c_prev = c_i

    self.module_logits = tf.stack(module_logit_list, axis=1)
    self.module_probs = tf.stack(module_prob_list)
    self.module_prob_list = module_prob_list
    self.c_list = c_list
    self.cv_list = cv_list
    self.c_st_list = c_st_list


class SubqController:

  def __init__(self, main_config, lstm_seq, q_encoding, embed_seq, seq_length_batch,
               is_train, scope='subq_controller', reuse=None, gt_module_ids=None):
    """
    This sub-question controller produces the sub-question context vector at every step.

    Input:
      lstm_seq: [N, S, d], tf.float32
      q_encoding: [N, d], tf.float32
      embed_seq: [N, S, e], tf.float32
      seq_length_batch: [N], tf.int32
    """
    self.is_train = is_train
    dim = cfg.MODEL.LSTM_DIM
    ctrl_dim = (cfg.MODEL.EMBED_DIM if cfg.MODEL.CTRL.USE_WORD_EMBED
          else cfg.MODEL.LSTM_DIM)
    T_ctrl = cfg.MODEL.T_CTRL
    
    # an attention mask to normalize textual attention over the actual
    # sequence length
    lstm_seq = tf.transpose(lstm_seq, perm=[1, 0, 2]) # Transpose to [S, N, d]
    embed_seq = tf.transpose(embed_seq, perm=[1, 0, 2])
    S = tf.shape(lstm_seq)[0]
    N = tf.shape(lstm_seq)[1]
    # att_mask: [S, N, 1]
    att_mask = tf.less(tf.range(S)[:, ax, ax], seq_length_batch[:, ax])
    att_mask = tf.cast(att_mask, tf.float32)
    with tf.variable_scope(scope, reuse=reuse):
      S = tf.shape(lstm_seq)[0]
      N = tf.shape(lstm_seq)[1]

      # manually unrolling for a number of timesteps
      c_init = tf.get_variable(
        'c_init', [1, ctrl_dim],
        initializer=tf.initializers.random_normal(
          stddev=np.sqrt(1. / ctrl_dim)))
      c_prev = tf.tile(c_init, to_T([N, 1]))
      c_prev.set_shape([None, ctrl_dim])
      c_list = []
      cv_list = []
      c_st_list = []
      module_logit_list = []
      module_prob_list = []
      for t in range(T_ctrl):
        q_i = fc('fc_q_%d' % t, q_encoding, output_dim=dim, is_train=self.is_train, dropout=main_config.nmn_dropout)  # [N, d]  
        q_i_c = tf.concat([q_i, c_prev], axis=1)  # [N, 2d]
        cq_i = fc('fc_cq', q_i_c, output_dim=dim, is_train=self.is_train, reuse=(t > 0), dropout=main_config.nmn_dropout)
        c_st_list.append(cq_i)

        elem_prod = tf.reshape(cq_i * lstm_seq, to_T([S*N, dim]))
        elem_prod.set_shape([None, dim])  # [S*N, d]
        raw_cv_i = tf.reshape(
          fc('fc_cv_i', elem_prod, output_dim=1, is_train=self.is_train, reuse=(t > 0), dropout=main_config.nmn_dropout),
          to_T([S, N, 1]))
        cv_i = tf.nn.softmax(raw_cv_i, axis=0)  # [S, N, 1]
        # normalize the attention over the actual sequence length
        if cfg.MODEL.CTRL.NORMALIZE_ATT:
          cv_i = cv_i * att_mask
          cv_i /= tf.reduce_sum(cv_i, 0, keepdims=True)

        if cfg.MODEL.CTRL.USE_WORD_EMBED:
          c_i = tf.reduce_sum(cv_i * embed_seq, axis=[0])  # [N, e]
        else:
          c_i = tf.reduce_sum(cv_i * lstm_seq, axis=[0])  # [N, d]
        c_list.append(c_i)
        cv_list.append(cv_i)
        c_prev = c_i

    self.c_list = c_list
    self.cv_list = cv_list
    self.c_st_list = c_st_list


class ModuleController:

  def __init__( self, main_config, lstm_seq, q_encoding, embed_seq, seq_length_batch,
                num_module, is_train, scope='module_controller', reuse=None, gt_module_ids=None):
    """
    This module controller produces the module probablity at every step.

    Input:
      lstm_seq: [N, S, d], tf.float32
      q_encoding: [N, d], tf.float32
      embed_seq: [N, S, e], tf.float32
      seq_length_batch: [N], tf.int32
    """

    self.is_train = is_train
    dim = cfg.MODEL.LSTM_DIM
    ctrl_dim = (cfg.MODEL.EMBED_DIM if cfg.MODEL.CTRL.USE_WORD_EMBED
          else cfg.MODEL.LSTM_DIM)
    T_ctrl = cfg.MODEL.T_CTRL

    # an attention mask to normalize textual attention over the actual
    # sequence length
    lstm_seq = tf.transpose(lstm_seq, perm=[1, 0, 2]) # Transpose to [S, N, d]
    embed_seq = tf.transpose(embed_seq, perm=[1, 0, 2])
    S = tf.shape(lstm_seq)[0]
    N = tf.shape(lstm_seq)[1]
    # att_mask: [S, N, 1]
    att_mask = tf.less(tf.range(S)[:, ax, ax], seq_length_batch[:, ax])
    att_mask = tf.cast(att_mask, tf.float32)
    with tf.variable_scope(scope, reuse=reuse):
      S = tf.shape(lstm_seq)[0]
      N = tf.shape(lstm_seq)[1]

      # manually unrolling for a number of timesteps
      c_init = tf.get_variable(
        'c_init', [1, ctrl_dim],
        initializer=tf.initializers.random_normal(
          stddev=np.sqrt(1. / ctrl_dim)))
      c_prev = tf.tile(c_init, to_T([N, 1]))
      c_prev.set_shape([None, ctrl_dim])
      c_list = []
      cv_list = []
      c_st_list = []
      module_logit_list = []
      module_prob_list = []
      for t in range(T_ctrl):
        q_i = fc('fc_q_%d' % t, q_encoding, output_dim=dim, is_train=self.is_train, dropout=main_config.nmn_dropout)  # [N, d]
        q_i_c = tf.concat([q_i, c_prev], axis=1)  # [N, 2d]
        cq_i = fc('fc_cq', q_i_c, output_dim=dim, is_train=self.is_train, reuse=(t > 0), dropout=main_config.nmn_dropout)
        c_st_list.append(cq_i)
        if cfg.MODEL.CTRL.LINEAR_MODULE_WEIGHTS:
          # Apply a linear transform on cq_i to predict the module
          # weights
          module_logit = fc(
            'fc_module_w_linear', cq_i, output_dim=num_module, is_train=self.is_train,
            bias_term=False, reuse=(t > 0), dropout=main_config.nmn_dropout)  # [N, M]
        else:
          # Apply a fully connected network on top of cq_i to predict
          # the module weights
          bias_term = cfg.MODEL.CTRL.MLP_MODULE_WEIGHTS_BIAS_TERM
          module_w_l1 = fc_elu(
            'fc_module_w_layer1', cq_i, output_dim=dim, is_train=self.is_train,
            reuse=(t > 0), dropout=main_config.nmn_dropout)
          module_logit = fc(
            'fc_module_w_layer2', module_w_l1,
            output_dim=num_module, is_train=self.is_train, bias_term=bias_term,
            reuse=(t > 0), dropout=main_config.nmn_dropout)  # [N, M]
        module_logit_list.append(module_logit)
        if cfg.MODEL.CTRL.USE_GUMBEL_SOFTMAX:
          module_prob = gumbel_softmax(
            module_logit, cfg.MODEL.CTRL.GUMBEL_SOFTMAX_TMP)
        else:
          module_prob = tf.nn.softmax(module_logit, axis=1)
        # Use hard (discrete) layout if specified
        if cfg.MODEL.CTRL.USE_HARD_ARGMAX_LAYOUT:
          if t == 0:
            print('Using hard argmax layout. '
                'This should only be used at test time!')
          module_prob = tf.one_hot(
            tf.argmax(module_prob, axis=-1), num_module,
            dtype=module_prob.dtype)
        module_prob_list.append(module_prob)

        elem_prod = tf.reshape(cq_i * lstm_seq, to_T([S*N, dim]))
        elem_prod.set_shape([None, dim])  # [S*N, d]
        raw_cv_i = tf.reshape(
          fc('fc_cv_i', elem_prod, output_dim=1, is_train=self.is_train, reuse=(t > 0), dropout=main_config.nmn_dropout),
          to_T([S, N, 1]))
        cv_i = tf.nn.softmax(raw_cv_i, axis=0)  # [S, N, 1]
        # normalize the attention over the actual sequence length
        if cfg.MODEL.CTRL.NORMALIZE_ATT:
          cv_i = cv_i * att_mask
          cv_i /= tf.reduce_sum(cv_i, 0, keepdims=True)

        if cfg.MODEL.CTRL.USE_WORD_EMBED:
          c_i = tf.reduce_sum(cv_i * embed_seq, axis=[0])  # [N, e]
        else:
          c_i = tf.reduce_sum(cv_i * lstm_seq, axis=[0])  # [N, d]
        c_list.append(c_i)
        cv_list.append(cv_i)
        c_prev = c_i

    self.module_logits = tf.stack(module_logit_list, axis=1)
    self.module_probs = tf.stack(module_prob_list)
    self.module_prob_list = module_prob_list