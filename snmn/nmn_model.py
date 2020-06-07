import tensorflow as tf
from tensorflow import convert_to_tensor as to_T

from snmn.config import cfg
from snmn import controller, nmn


class NMN_Model:
  def __init__(self, config, lstm_seq, embed_seq, q_encoding, seq_mask_batch, 
               seq_length_batch, kb_batch, kb_mask_batch, kb_len_batch, module_names, 
               is_training, 
               scope='nmn_model', reuse=None, max_para_size=None):
    """
    Neual Module Networks v4 (the whole model)

    Input:
      lstm_seq: [N, q_len, 2d], tf.float32
      embed_seq: [N, q_len, embdim], tf.float32
      seq_length_batch: [N], tf.int32
      kb_batch: [N, num_para, para_len, 2d], tf.float32
    """

    with tf.variable_scope(scope, reuse=reuse):
      self.T_ctrl = cfg.MODEL.T_CTRL
      self.config = config

      # Controller and NMN
      num_module = len(module_names)
      self.num_module = num_module

      if config.nmn_separate_controllers:
        self.module_controller = controller.ModuleController(config, 
                                                            tf.stop_gradient(lstm_seq), 
                                                            tf.stop_gradient(q_encoding),
                                                            tf.stop_gradient(embed_seq), 
                                                            seq_length_batch,
                                                            num_module, 
                                                            is_training)
        self.module_logits = self.module_controller.module_logits
        self.module_probs = self.module_controller.module_probs
        self.module_prob_list = self.module_controller.module_prob_list


        self.subq_controller = controller.SubqController(config, lstm_seq, q_encoding, \
                                                        embed_seq, seq_length_batch, is_training)
        self.c_list = self.subq_controller.c_list
        self.cv_list = self.subq_controller.cv_list
        self.c_st_list = self.subq_controller.c_st_list
        self.c_last = self.subq_controller.c_list[-1]
      else:
        self.controller = controller.Controller(config, lstm_seq, q_encoding, \
                                                embed_seq, seq_length_batch, \
                                                num_module, is_training)
        self.c_list = self.controller.c_list
        self.cv_list = self.controller.cv_list
        self.c_st_list = self.controller.c_st_list
        self.module_logits = self.controller.module_logits
        self.module_probs = self.controller.module_probs
        self.module_prob_list = self.controller.module_prob_list
        self.c_last = self.controller.c_list[-1]
      
      self.nmn = nmn.NMN(config, kb_batch, kb_mask_batch, kb_len_batch,
                         self.c_list, self.cv_list, lstm_seq, seq_mask_batch, 
                         module_names, self.module_prob_list, is_training, seq_length_batch, 
                         max_para_size=max_para_size)

      self.mem_last = self.nmn.mem_last
      self.att_last = self.nmn.att_last
      self.att_first = self.nmn.att_first
      self.att_second = self.nmn.att_second
      
      if self.config.supervise_bridge_entity:
        self.bridge_logits = self.nmn.bridge_logits

      self.att_in_1 = self.nmn.att_in_1
      self.att_in_2 = self.nmn.att_in_2

      self.params = [
        v for v in tf.trainable_variables() if scope in v.op.name]
      

  def sharpen_loss(self):
    # module_probs has shape [T, N, M]
    # flat_probs has shape [T*N, M]
    flat_probs = tf.reshape(self.module_probs, [-1, self.num_module])
    loss_type = cfg.TRAIN.SHARPEN_LOSS_TYPE
    if loss_type == 'max_prob':
      # the difference between the maximum weight (probability) and 1
      margin = 1 - tf.reduce_max(flat_probs, axis=-1)
      sharpen_loss = tf.reduce_mean(margin)
    elif loss_type == 'entropy':
      # the entropy of the module weights
      entropy = -tf.reduce_sum(
        tf.log(tf.maximum(flat_probs, 1e-5)) * flat_probs, axis=-1)
      sharpen_loss = tf.reduce_mean(entropy)
    else:
      raise ValueError(
        'Unknown layout sharpen loss type: {}'.format(loss_type))
    return sharpen_loss
