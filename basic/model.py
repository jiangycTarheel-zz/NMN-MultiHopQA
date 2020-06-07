import random
import os
import itertools
import tensorflow as tf
from tensorflow import newaxis as ax

from basic.attention_modules import hotpot_biattention, zhong_selfatt
from basic.batcher import get_batch_feed_dict
from my.tensorflow import get_initializer
from my.tensorflow.nn import softsel, get_logits, linear_logits, highway_network, multi_conv1d, dense
from my.tensorflow.ops import bi_cudnn_rnn_encoder
from snmn.nmn_model import NMN_Model


def get_multi_gpu_models(config, emb_mat=None):
  models = []
  with tf.variable_scope(tf.get_variable_scope()) as vscope:
    for gpu_idx in range(config.num_gpus):
      with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
        if gpu_idx > 0:
          tf.get_variable_scope().reuse_variables()
        model = Model(config, scope, emb_mat, rep=gpu_idx == 0)
        models.append(model)
  return models


class Model(object):
  def __init__(self, config, scope, emb_mat, rep=True):
    self.scope = scope
    self.config = config
    self.emb_mat = emb_mat
    self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                       initializer=tf.constant_initializer(0), trainable=False)
    N, M, JX, JQ, VW, VC, W = \
      config.batch_size, config.max_num_sents, config.max_sent_size, \
      config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
    self.x = tf.placeholder('int32', [N, None, None], name='x')
    self.cx = tf.placeholder('int32', [N, None, None, W], name='cx')
    self.x_mask = tf.placeholder('bool', [N, None, None], name='x_mask')

    if config.dataset == 'hotpotqa':
      self.q_type_labels = tf.placeholder('int32', [N, None], name='q_type_labels')
      self.q_yesno_labels = tf.placeholder('int32', [N, None], name='q_yesno_labels')
      self.yes_no = tf.placeholder('bool', [N], name='yes_no')

    self.max_para_size = tf.placeholder('int32', [], name='max_para_size')
    self.q = tf.placeholder('int32', [N, None], name='q')
    self.cq = tf.placeholder('int32', [N, None, W], name='cq')
    self.q_mask = tf.placeholder('bool', [N, None], name='q_mask')
    self.y = tf.placeholder('bool', [N, None, None], name='y')
    self.y2 = tf.placeholder('bool', [N, None, None], name='y2')
    self.wy = tf.placeholder('bool', [N, None, None], name='wy')
    self.is_train = tf.placeholder('bool', [], name='is_train')
    self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')
    self.na = tf.placeholder('bool', [N], name='na')

    if config.supervise_bridge_entity:
      self.bridge_word_in_context_ids = tf.placeholder('int32', [N, None], name='bridge_word_in_context_ids')
      self.bridge_na = tf.placeholder('bool', [N], name='bridge_na')

    # if config.reasoning_layer == 'snmn':
    #   self.module_prob_feed = tf.placeholder('float32', [3, N, 4], name='module_prob_feed')

    # Define misc
    self.tensor_dict = {}

    # Forward outputs / loss inputs
    self.logits = None
    self.yp = None
    self.var_list = None
    self.na_prob = None

    # Loss outputs
    self.loss = None

    self._build_forward()
    self._build_loss()
    self.var_ema = None
    if rep:
      self._build_var_ema()
      if config.mode == 'train':
        self._build_ema()

    self.summary = tf.summary.merge_all()
    self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))


  def _build_forward(self):
    config = self.config

    x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
    q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]
  
    N, M, JX, JQ, VW, VC, d, W = \
      config.batch_size, config.max_num_sents, config.max_sent_size, \
      config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
      config.max_word_size

    JX = tf.shape(self.x)[2]
    JQ = tf.shape(self.q)[1]
    M = tf.shape(self.x)[1]
    dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

    with tf.variable_scope("emb"):
      if config.use_char_emb:
        with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
          char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

        with tf.variable_scope("char"):
          Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
          Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]  
          Acx = tf.reshape(Acx, [-1, JX, W, dc])
          Acq = tf.reshape(Acq, [-1, JQ, W, dc])
          
          filter_sizes = list(map(int, config.out_channel_dims.split(',')))
          heights = list(map(int, config.filter_heights.split(',')))
          assert sum(filter_sizes) == dco, (filter_sizes, dco)
          with tf.variable_scope("conv"):
            xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
            if config.share_cnn_weights:
              tf.get_variable_scope().reuse_variables()
              qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
            else:
              qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")                
            xx = tf.reshape(xx, [-1, M, JX, dco])
            qq = tf.reshape(qq, [-1, JQ, dco])

      if config.use_word_emb:
        with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
          if config.mode == 'train':
            word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(self.emb_mat))
          else:
            word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
          if config.use_glove_for_unk:
            word_emb_mat = tf.concat(axis=0, values=[word_emb_mat, self.new_emb_mat])

        with tf.name_scope("word"):
          Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
          Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]
          self.tensor_dict['x'] = Ax
          self.tensor_dict['q'] = Aq

        if config.use_char_emb:
          xx = tf.concat(axis=3, values=[xx, Ax])  # [N, M, JX, di]
          qq = tf.concat(axis=2, values=[qq, Aq])  # [N, JQ, di]          
        else:
          xx = Ax
          qq = Aq

    # highway network
    if config.highway:
      with tf.variable_scope("highway"):
        xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train, input_keep_prob=config.highway_keep_prob)
        tf.get_variable_scope().reuse_variables()
        qq = highway_network(qq, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train, input_keep_prob=config.highway_keep_prob)
        
    self.tensor_dict['xx'] = xx
    self.tensor_dict['qq'] = qq
    
    with tf.variable_scope("prepro"):
      with tf.variable_scope('u1'):
        u, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, qq, q_len, self.is_train)
        if config.reasoning_layer == 'snmn':
          u_st = zhong_selfatt(u[:, ax, :, :], config.hidden_size*2, seq_len=q_len, transform='squeeze')
      
      if config.share_lstm_weights:
        with tf.variable_scope('u1', reuse=True):
          h, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, tf.squeeze(xx, axis=1), tf.squeeze(x_len, axis=1), self.is_train)
          h = h[:, ax, :, :]
      else:
        with tf.variable_scope('h1'):
          h, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, tf.squeeze(xx, axis=1), tf.squeeze(x_len, axis=1), self.is_train)
          h = h[:, ax, :, :]
    
      self.tensor_dict['u'] = u
      self.tensor_dict['h'] = h

    with tf.variable_scope("main"):
      context_dim = config.hidden_size * 2
      ### Reconstruct before bidaf because otherwise we need to build a larger query tensor.

      x_mask = self.x_mask
      x_len_squeeze = tf.squeeze(x_len, axis=1)
      p0 = h      
      
      ### Main model 
      if config.reasoning_layer == 'snmn':
        module_names = ['_Find', '_Compare', '_Relocate', '_NoOp']

        self.snmn = NMN_Model(config, u, qq, u_st, self.q_mask, q_len, p0, x_mask, x_len, module_names, \
                              self.is_train)
        self.u_weights = self.snmn.cv_list # question word distribution at each step
        self.module_prob_list = self.snmn.module_prob_list # module probability at each step

        g0 = tf.squeeze(self.snmn.att_second, axis=-1)

        if config.supervise_bridge_entity:
          self.hop0_logits = self.snmn.bridge_logits

        if config.self_att:
          with tf.variable_scope('g0'):
            g0, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, tf.squeeze(g0, axis=1), x_len_squeeze, self.is_train)
            g0 = g0[:, ax, :, :]
            g0 = hotpot_biattention(config, self.is_train, g0, tf.squeeze(g0, axis=1), h_mask=x_mask, u_mask=tf.squeeze(x_mask, axis=1), scope="self_att", tensor_dict=self.tensor_dict)
          g0 = tf.layers.dense(g0, config.hidden_size*2)

        with tf.variable_scope('g1'):
          g1, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob, tf.squeeze(g0, axis=1), tf.squeeze(x_len, axis=1), self.is_train)
          g1 = g1[:, ax, :, :]

        logits = get_logits([g1, g0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
                            mask=x_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')

        with tf.variable_scope('g2'):
          a1i = softsel(tf.reshape(g1, [N, M * JX, 2 * d]), tf.reshape(logits, [N, M * JX]))
          a1i = tf.tile(a1i[:, ax, ax, :], [1, M, JX, 1])
          g2, _ = bi_cudnn_rnn_encoder('lstm', config.hidden_size, 1, 1-config.input_keep_prob,
                                       tf.squeeze(tf.concat(axis=3, values=[g0, g1, a1i, g0 * a1i]), axis=1), 
                                       x_len_squeeze, self.is_train)
          g2 = g2[:, ax, :, :]
        logits2 = get_logits([g2, g1], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob, 
                              mask=x_mask, is_train=self.is_train, func=config.answer_func, 
                              scope='logits2')
        
        if config.dataset == 'hotpotqa':
          with tf.variable_scope('g3'):
            if config.nmn_qtype_class == 'mem_last':
              g3 = tf.concat([self.snmn.mem_last[:, ax, :], u_st[:, ax, :]], axis=-1)
            elif config.nmn_qtype_class == 'ctrl_st':
              g3 = self.snmn.c_st_list[0][:, ax, :]
            else:
              raise NotImplementedError

            self.predict_type = dense(g3, 2, scope='predict_type')
            g3_1 = self.snmn.mem_last[:, ax, :]
            self.predict_yesno = dense(g3_1, 2, scope='predict_yesno')

      flat_logits = tf.reshape(logits, [-1, M * JX])
      flat_yp = tf.nn.softmax(flat_logits)  # [-1, M * JX]
      flat_logits2 = tf.reshape(logits2, [-1, M * JX])
      flat_yp2 = tf.nn.softmax(flat_logits2)
      yp = tf.reshape(flat_yp, [-1, M, JX])
      yp2 = tf.reshape(flat_yp2, [-1, M, JX])
      wyp = tf.nn.sigmoid(logits2)
      self.logits = flat_logits
      self.logits2 = flat_logits2
      self.yp = yp
      self.yp2 = yp2
      self.wyp = wyp

      if config.dataset == 'hotpotqa':
        flat_predict_type = tf.reshape(self.predict_type, [-1, 2])
        flat_yp3 = tf.nn.softmax(flat_predict_type)
        self.yp3 = tf.reshape(flat_yp3, [-1, 1, 2])

        flat_predict_yesno = tf.reshape(self.predict_yesno, [-1, 2])
        flat_yp3_yesno = tf.nn.softmax(flat_predict_yesno)
        self.yp3_yesno = tf.reshape(flat_yp3_yesno, [-1, 1, 2])


  def _build_loss(self):
    config = self.config
    M = tf.shape(self.x)[1]
    JX = tf.shape(self.x)[2]

    # loss_mask will mask out hotpotqa examples with yes/no type answer.
    loss_mask = tf.logical_and(tf.cast(tf.reduce_max(tf.cast(self.q_mask, 'float'), 1), 'bool'), tf.logical_not(self.na))
    if config.supervise_bridge_entity:
      bridge_loss_mask = tf.cast(tf.logical_and(loss_mask, tf.logical_not(self.bridge_na)), 'float')
    if config.dataset == 'hotpotqa':
      yesno_mask = tf.cast(tf.logical_and(loss_mask, self.yes_no), 'float')
      loss_mask = tf.logical_and(loss_mask, tf.logical_not(self.yes_no))
      
    loss_mask = tf.cast(loss_mask, 'float')
    q_loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)

    losses = tf.nn.softmax_cross_entropy_with_logits(
      logits=self.logits, labels=tf.cast(tf.reshape(self.y, [-1, M * JX]), 'float'))
    losses2 = tf.nn.softmax_cross_entropy_with_logits(
      logits=self.logits2, labels=tf.cast(tf.reshape(self.y2, [-1, M * JX]), 'float'))

    if config.dataset == 'hotpotqa':
      losses_type = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_type, labels=self.q_type_labels)
      ce_loss_type = tf.reduce_mean(q_loss_mask * losses_type, name='loss_q_type')
      tf.summary.scalar(ce_loss_type.op.name, ce_loss_type)
      tf.add_to_collection('ema/scalar', ce_loss_type)
      tf.add_to_collection("losses", ce_loss_type)

      losses_yesno = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predict_yesno, labels=self.q_yesno_labels)
      ce_loss_yesno = tf.reduce_mean(yesno_mask * losses_yesno, name='loss_q_yesno') * config.yesno_loss_coeff
      tf.summary.scalar(ce_loss_yesno.op.name, ce_loss_yesno)
      tf.add_to_collection('ema/scalar', ce_loss_yesno)
      tf.add_to_collection("losses", ce_loss_yesno)

      ce_loss = tf.reduce_mean(loss_mask * losses)
      ce_loss2 = tf.reduce_mean(loss_mask * losses2)
      tf.add_to_collection('losses', ce_loss)
      tf.add_to_collection("losses", ce_loss2)
  
    self.loss = tf.add_n(tf.get_collection('losses', scope=self.scope), name='loss')
    tf.summary.scalar(self.loss.op.name, self.loss)
    tf.add_to_collection('ema/scalar', self.loss)

    if config.supervise_bridge_entity:
      bridge_word_ids = tf.squeeze(tf.slice(self.bridge_word_in_context_ids, [0, 0], [-1, 1]), axis=1)
      hop0_attn_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.hop0_logits, labels=bridge_word_ids)
      hop0_attn_loss = tf.reduce_mean(hop0_attn_losses * bridge_loss_mask, name='hop0_attn_loss')
      tf.summary.scalar('hop0_attn_loss', hop0_attn_loss)
      tf.add_to_collection('ema/scalar', hop0_attn_loss)
      self.loss += config.hop0_attn_loss_coeff * hop0_attn_loss
    
    tf.summary.scalar('total_loss', self.loss)


  def _build_ema(self):
    self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
    ema = self.ema
    tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
    ema_op = ema.apply(tensors)
    for var in tf.get_collection("ema/scalar", scope=self.scope):
      ema_var = ema.average(var)
      tf.summary.scalar(ema_var.op.name, ema_var)
    for var in tf.get_collection("ema/vector", scope=self.scope):
      ema_var = ema.average(var)
      tf.summary.histogram(ema_var.op.name, ema_var)
    with tf.control_dependencies([ema_op]):
      self.loss = tf.identity(self.loss)


  def _build_var_ema(self):
    self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
    ema = self.var_ema
    ema_op = ema.apply(tf.trainable_variables())      
    with tf.control_dependencies([ema_op]):
      self.loss = tf.identity(self.loss)

  
  def get_loss(self):
    return self.loss


  def get_global_step(self):
    return self.global_step


  def get_var_list(self, model_name):
    if model_name == 'model_network':
      var_list = [var for var in tf.trainable_variables() if 'reward_network' not in var.name and 'ranker' not in var.name and 'controller' not in var.name]
    elif model_name == 'controller':
      var_list = [var for var in tf.trainable_variables() if 'module_controller' in var.name]
    elif model_name == 'nmn':
      var_list = [var for var in tf.trainable_variables() if 'module_controller' not in var.name]
    elif model_name == 'all':
      var_list = [var for var in tf.trainable_variables()]
    else:
      raise NotImplementedError
    
    assert len(var_list) > 0
    return var_list


  def get_feed_dict(model, batch, is_train, supervised=True):
    return get_batch_feed_dict(model, batch, is_train, supervised=True)