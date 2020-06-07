import numpy as np
import tensorflow as tf
from tensorflow import convert_to_tensor as to_T, newaxis as ax
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from my.tensorflow.ops import bi_cudnn_rnn_encoder

from snmn.config import cfg
from snmn.util.cnn import fc_layer as fc, conv_layer as conv
from basic.attention_modules import weighted_biattention
from my.tensorflow.nn import dense
from my.tensorflow.nn import linear_logits


MODULE_INPUT_NUM = {
  '_NoOp': 2,
  '_Find': 0,
  '_Relocate': 1,
  '_Compare': 2,
}

MODULE_OUTPUT_NUM = {
  '_NoOp': 2,
  '_Find': 1,
  '_Relocate': 1,
  '_Compare': 1,
}


class NMN:
  def __init__(self, config, kb_batch, kb_mask_batch, kb_len_batch, c_list, 
               cv_list, q, q_mask, module_names, module_prob_list, is_train, 
               q_len, max_para_size=None, scope='NMN', reuse=None):
    """
    NMN v4 with an attention stack
    """
    with tf.variable_scope(scope, reuse=reuse):
      self.is_train = is_train
      self.main_config = config
      self.kb_batch = kb_batch
      self.kb_mask_batch = kb_mask_batch
      self.kb_len_batch = kb_len_batch
      self.c_list = c_list
      self.cv_list = cv_list
      self.q = q
      self.q_len = q_len
      self.q_mask = q_mask
      self.module_prob_list = module_prob_list
      self.kb_dim = cfg.MODEL.KB_DIM
      if self.main_config.nmn_relocate_move_ptr:
        MODULE_OUTPUT_NUM['_Relocate'] = MODULE_INPUT_NUM['_Relocate'] + 1
      else:
        MODULE_OUTPUT_NUM['_Relocate'] = MODULE_INPUT_NUM['_Relocate']
      
      self.T_ctrl = cfg.MODEL.T_CTRL
      self.mem_dim = cfg.MODEL.NMN.MEM_DIM
      self.N = tf.shape(kb_batch)[0]
      self.M = tf.shape(kb_batch)[1]
      self.JX = tf.shape(kb_batch)[2]
      self.att_shape = to_T([self.N, self.M, self.JX, 1])
      self.max_para_size = max_para_size
      
      self.stack_len = cfg.MODEL.NMN.STACK.LENGTH
      # The initialial stack values are all zeros everywhere
      self.att_stack_init = tf.zeros(
        to_T([self.N, self.M, self.JX, self.kb_dim, self.stack_len]))
      # The initial stack pointer points to the stack bottom
      self.stack_ptr_init = tf.one_hot(
        tf.zeros(to_T([self.N]), tf.int32), self.stack_len)
      self.mem_init = tf.zeros(to_T([self.N, self.mem_dim]))

      # zero-outputs that can be easily used by the modules
      self.att_zero = tf.zeros(self.att_shape, tf.float32)
      if config.nmn_mem_init == 'zero':
        self.mem_zero = tf.zeros(to_T([self.N, self.mem_dim]), tf.float32)
      elif config.nmn_mem_init == 'random':
        self.mem_zero = tf.random.uniform(to_T([self.N, self.mem_dim]), minval=-100, maxval=100, dtype=tf.float32)
      else:
        raise NotImplementedError
      self.score_zero = tf.zeros(to_T([self.N, 1]), tf.float32)

      # the set of modules and functions (e.g. "_Find" -> Find)
      self.module_names = module_names
      self.module_funcs = [getattr(self, m[1:]) for m in module_names]
      self.num_module = len(module_names)
      self.module_validity_mat = _build_module_validity_mat(module_names)

      # unroll the modules with a fixed number of timestep T_ctrl
      self.att_list = []
      self.att_stack_list = []
      self.stack_ptr_list = []
      self.mem_list = []
      self.score_list = []
      self.att_in_1, self.att_in_2 = None, None
      att_stack_prev = self.att_stack_init
      stack_ptr_prev = self.stack_ptr_init
      mem_prev = self.mem_init
      
      self.module_validity = []
      for t in range(self.T_ctrl):
        c_i = self.c_list[t]
        cv_i = self.cv_list[t]
        module_prob = self.module_prob_list[t]
        # only keep the prob of valid modules (i.e. those won't cause
        # stack underflow or overflow. e.g. _Filter can't be run at
        # t = 0 since the stack is empty).
        if cfg.MODEL.NMN.VALIDATE_MODULES:
          module_validity = tf.matmul(
            stack_ptr_prev, self.module_validity_mat)
          if cfg.MODEL.NMN.HARD_MODULE_VALIDATION:
            module_validity = tf.round(module_validity)
          self.module_validity.append(module_validity)
          module_prob *= module_validity
          module_prob /= tf.reduce_sum(
            module_prob, axis=1, keepdims=True)
          self.module_prob_list[t] = module_prob

        # run all the modules, and average their results wrt module_w
        res = [f(att_stack_prev, stack_ptr_prev, mem_prev, c_i, cv_i, 
                t, reuse=(t > 0)) for f in self.module_funcs]

        att_stack_avg = tf.reduce_sum(
          module_prob[:, ax, ax, ax, ax, :] *
          tf.stack([r[0] for r in res], axis=-1), axis=-1)
        stack_ptr_avg = _sharpen_ptr(tf.reduce_sum(
          module_prob[:, ax, :] *
          tf.stack([r[1] for r in res], axis=2), axis=-1))
        mem_avg = tf.reduce_sum(
          module_prob[:, ax, :] *
          tf.stack([r[2] for r in res], axis=2), axis=-1)
        score_avg = tf.reduce_sum(
          module_prob[:, ax, :] * 
          tf.stack([r[3] for r in res], axis=2), axis=-1)

        self.att_list.append(_read_from_stack(att_stack_avg, stack_ptr_avg))
        self.att_stack_list.append(att_stack_avg)
        self.stack_ptr_list.append(stack_ptr_avg)
        self.mem_list.append(mem_avg)
        self.score_list.append(score_avg)
        att_stack_prev = att_stack_avg
        stack_ptr_prev = stack_ptr_avg
        mem_prev = mem_avg

      self.att_last = self.att_list[-1]
      self.att_first = self.att_list[0]
      self.att_second = self.att_list[1]
      self.mem_last = self.mem_list[-1]
      self.score_last = self.score_list[-1]


  def NoOp(self, att_stack, stack_ptr, mem_in, c_i, cv_i, t, scope='NoOp', reuse=None):
    """
    Does nothing. It leaves the stack pointer, the stack and mem vector
    as-is.
    """
    return att_stack, stack_ptr, mem_in, self.score_zero


  def Find(self, att_stack, stack_ptr, mem_in, c_i, cv_i, t, scope='Find', reuse=None):
    """
    Performs localization, and updates memory vector.
    """
    with tf.variable_scope(scope, reuse=reuse):
      # Get attention
      #   1) linearly map the controller vectors to the KB dimension
      #   2) elementwise product with KB
      #   3) 1x1 convolution to get attention logits
      c_mapped = fc('fc_c_mapped', c_i, output_dim=cfg.MODEL.KB_DIM, \
          is_train=self.is_train, dropout=self.main_config.nmn_dropout)

      kb_batch = self.kb_batch

      elt_prod = tf.nn.l2_normalize(
          kb_batch * c_mapped[:, ax, ax, :], axis=-1)

      att_out = self.apply_attention(elt_prod, cv_i, t)

      # Push to stack
      stack_ptr = _move_ptr_fw(stack_ptr)
      att_stack = _write_to_stack(att_stack, stack_ptr, att_out)
      
      # Find bridge entity
      if  t == 0 and (self.main_config.supervise_bridge_entity):
        g1, _ = bi_cudnn_rnn_encoder('lstm', self.main_config.hidden_size, 1, 1-self.main_config.input_keep_prob, \
                                    tf.squeeze(att_out, axis=1), tf.squeeze(self.kb_len_batch, axis=1), self.is_train)
        g1 = g1[:, ax, :, :]
        bridge_logits = linear_logits([g1, att_out], True, input_keep_prob=self.main_config.input_keep_prob, \
                                     mask=self.kb_mask_batch, is_train=self.is_train, scope='logits_bridge')
        self.bridge_logits = tf.squeeze(bridge_logits, axis=1)
        bridge_attn = tf.nn.softmax(self.bridge_logits)
        self.bridge_attn = bridge_attn
        new_mem = tf.einsum('ijk,ij->ik', tf.squeeze(kb_batch, axis=1), bridge_attn)
        if t == 0:
          self.inferred_bridge = new_mem

    return att_stack, stack_ptr, self.mem_zero, self.score_zero


  def Compare(self, att_stack, stack_ptr, mem_in, c_i, cv_i, t, scope='Compare', reuse=None):
    # Pop from stack
    att_in_2 = tf.squeeze(_read_from_stack(att_stack, stack_ptr), axis=-1)
    stack_ptr = _move_ptr_bw(stack_ptr)
    att_in_1 = tf.squeeze(_read_from_stack(att_stack, stack_ptr), axis=-1)
    #stack_ptr = _move_ptr_bw(stack_ptr) 
    stack_ptr = _move_ptr_fw(stack_ptr)
    
    with tf.variable_scope(scope, reuse=reuse):

      att_prob_in_1 = linear_logits([att_in_1], True, input_keep_prob=self.main_config.input_keep_prob, \
                                mask=self.kb_mask_batch, is_train=self.is_train)
      att_prob_in_2 = linear_logits([att_in_2], True, input_keep_prob=self.main_config.input_keep_prob, \
                                mask=self.kb_mask_batch, is_train=self.is_train, reuse=True)
      att_prob_in_1, att_prob_in_2 = tf.squeeze(att_prob_in_1, axis=1), tf.squeeze(att_prob_in_2, axis=1) 

      self.att_in_1 = att_prob_in_1
      self.att_in_2 = att_prob_in_2
      c_mapped = fc('fc_c_mapped', c_i, output_dim=cfg.MODEL.KB_DIM, is_train=self.is_train, dropout=self.main_config.nmn_dropout)
      kb_att_in_1 = _extract_softmax_avg(self.kb_batch, att_prob_in_1[:, ax, :])
      kb_att_in_2 = _extract_softmax_avg(self.kb_batch, att_prob_in_2[:, ax, :])

      fc_in_1 = tf.concat([c_mapped, c_mapped * kb_att_in_1, c_mapped*kb_att_in_2, kb_att_in_1-kb_att_in_2], axis=1)
      mem_out = tf.nn.tanh(fc('fc_mem_out_1', fc_in_1, output_dim=self.mem_dim, is_train=self.is_train, dropout=self.main_config.nmn_dropout))
     
    return att_stack, stack_ptr, mem_out, self.score_zero


  def Relocate(self, att_stack, stack_ptr, mem_in, c_i, cv_i, t, scope='Transform', reuse=None):
    """
    Relocates the previous attention, and updates memory vector.
    """

    kb_batch = self.kb_batch

    with tf.variable_scope(scope, reuse=reuse):
      c_mapped = fc('fc_c_mapped', c_i, output_dim=cfg.MODEL.KB_DIM, is_train=self.is_train, dropout=self.main_config.nmn_dropout)

      # Get attention
      #   1) linearly map the controller vectors to the KB dimension
      #   2) extract attended features from the input attention
      #   2) elementwise product with KB
      #   3) 1x1 convolution to get attention logits

      if t == 0:
        elt_prod = tf.nn.l2_normalize(
          kb_batch * c_mapped[:, ax, ax, :], axis=-1)
      else:
        elt_prod = tf.nn.l2_normalize(
          kb_batch * c_mapped[:, ax, ax, :] *
          self.inferred_bridge[:, ax, ax, :], axis=-1)

      att_out = self.apply_attention(elt_prod, cv_i, t, module='relocate')

      # Push to stack
      if self.main_config.nmn_relocate_move_ptr:
        stack_ptr = _move_ptr_fw(stack_ptr)  # cancel-out above
      att_stack = _write_to_stack(att_stack, stack_ptr, att_out)

    return att_stack, stack_ptr, self.mem_zero, self.score_zero


  def apply_attention(self, h, u_weights, t, module='find'):
    if self.main_config.nmn_attention_type == 'conv':
      out = _1x1conv('conv_att_out', h, output_dim=1)
      logits = apply_mask(out, self.kb_mask_batch)
    elif self.main_config.nmn_attention_type == 'biattn':
      out = self.bi_attention(h, u_weights, t, module=module)
    else:
      raise NotImplementedError
    return out


  def bi_attention(self, h, u_weights, t, module='find'):
    u_weights = tf.transpose(tf.squeeze(u_weights, axis=-1), perm=[1, 0])
    q, q_mask = self.q, self.q_mask

    p0, context_dim, weight_one = weighted_biattention(
        h, q, self.kb_dim, h_mask=self.kb_mask_batch, u_mask=tf.cast(q_mask, 'bool'), 
        u_weights=u_weights, scope='biattn')
    if module == 'find' and t == 1:
      self.weight_one = weight_one
    
    p0 = dense(p0, self.main_config.hidden_size*2, scope='biattn_dense')
    return p0


def _move_ptr_fw(stack_ptr):
  """
  Move the stack pointer forward (i.e. to push to stack).
  """
  # Note: in TF, conv1d is implemented as auto-correlation (instead of
  # mathmatical convolution), so no flipping of the filter.
  filter_fw = to_T(np.array([1, 0, 0], np.float32).reshape((3, 1, 1)))
  new_stack_ptr = tf.squeeze(
    tf.nn.conv1d(stack_ptr[..., ax], filter_fw, 1, 'SAME'), axis=[2])
  # when the stack pointer is already at the stack top, keep
  # the pointer in the same location (otherwise the pointer will be all zero)
  if cfg.MODEL.NMN.STACK.GUARD_STACK_PTR:
    stack_len = cfg.MODEL.NMN.STACK.LENGTH
    stack_top_mask = tf.one_hot(stack_len - 1, stack_len)
    new_stack_ptr += stack_top_mask * stack_ptr
  return new_stack_ptr


def _move_ptr_bw(stack_ptr):
  """
  Move the stack pointer backward (i.e. to pop from stack).
  """
  # Note: in TF, conv1d is implemented as auto-correlation (instead of
  # mathmatical convolution), so no flipping of the filter.
  filter_fw = to_T(np.array([0, 0, 1], np.float32).reshape((3, 1, 1)))
  new_stack_ptr = tf.squeeze(
    tf.nn.conv1d(stack_ptr[..., ax], filter_fw, 1, 'SAME'), axis=[2])
  # when the stack pointer is already at the stack bottom, keep
  # the pointer in the same location (otherwise the pointer will be all zero)
  if cfg.MODEL.NMN.STACK.GUARD_STACK_PTR:
    stack_len = cfg.MODEL.NMN.STACK.LENGTH
    stack_bottom_mask = tf.one_hot(0, stack_len)
    new_stack_ptr += stack_bottom_mask * stack_ptr
  return new_stack_ptr


def _read_from_stack(att_stack, stack_ptr):
  """
  Read the value at the given stack pointer.
  """
  stack_ptr_expand = stack_ptr[:, ax, ax, ax, :]
  # The stack pointer is a one-hot vector, so just do dot product
  att = tf.reduce_sum(att_stack * stack_ptr_expand, axis=-1, keepdims=True)
  return att


def _write_to_stack(att_stack, stack_ptr, att):
  """
  Write value 'att' into the stack at the given stack pointer. Note that the
  result needs to be assigned back to att_stack
  """
  stack_ptr_expand = stack_ptr[:, ax, ax, ax, :]
  att_stack = att[:, :, :, :, ax] * stack_ptr_expand + att_stack * (1 - stack_ptr_expand)
  return att_stack


def _sharpen_ptr(stack_ptr):
  """
  Sharpen the stack pointers into (nearly) one-hot vectors, using argmax
  or softmax. The stack values should always sum up to one for each instance.
  """
  hard = cfg.MODEL.NMN.STACK.USE_HARD_SHARPEN
  if hard:
    # hard (non-differentiable) sharpening with argmax
    new_stack_ptr = tf.one_hot(
      tf.argmax(stack_ptr, axis=1), tf.shape(stack_ptr)[1])
  else:
    # soft (differentiable) sharpening with softmax
    temperature = cfg.MODEL.NMN.STACK.SOFT_SHARPEN_TEMP
    new_stack_ptr = tf.nn.softmax(stack_ptr / temperature)
  return new_stack_ptr


def _1x1conv(name, bottom, output_dim, reuse=None):
  return conv(name, bottom, kernel_size=1, stride=1, output_dim=output_dim,
        reuse=reuse)


def apply_mask(x, mask):
  return x - 1e30 * (1. - tf.cast(mask[:, :, :, ax], 'float32'))


def _spatial_softmax(att_raw):
  att_shape = tf.shape(att_raw)
  N = att_shape[0]
  att_softmax = tf.nn.softmax(tf.reshape(att_raw, to_T([N, -1])), axis=1)
  att_softmax = tf.reshape(att_softmax, att_shape)
  return att_softmax


def _extract_softmax_avg(kb_batch, att_raw):
  att_softmax = _spatial_softmax(att_raw)[:, :, :, ax]
  return tf.reduce_sum(kb_batch * att_softmax, axis=[1, 2])


def _build_module_validity_mat(module_names):
  """
  Build a module validity matrix, ensuring that only valid modules will have
  non-zero probabilities. A module is only valid to run if there are enough
  attentions to be popped from the stack, and have space to push into
  (e.g. _Find), so that stack will not underflow or overflow by design.

  module_validity_mat is a stack_len x num_module matrix, and is used to
  multiply with stack_ptr to get validity boolean vector for the modules.
  """
  stack_len = cfg.MODEL.NMN.STACK.LENGTH
  module_validity_mat = np.zeros((stack_len, len(module_names)), np.float32)
  for n_m, m in enumerate(module_names):
    # a module can be run only when stack ptr position satisfies
    # (min_ptr_pos <= ptr <= max_ptr_pos), where max_ptr_pos is inclusive
    # 1) minimum position:
    #    stack need to have MODULE_INPUT_NUM[m] things to pop from
    min_ptr_pos = MODULE_INPUT_NUM[m]
    # the stack ptr diff=(MODULE_OUTPUT_NUM[m] - MODULE_INPUT_NUM[m])
    # ensure that ptr + diff <= stack_len - 1 (stack top)
    max_ptr_pos = (
      stack_len - 1 + MODULE_INPUT_NUM[m] - MODULE_OUTPUT_NUM[m])
    module_validity_mat[min_ptr_pos:max_ptr_pos+1, n_m] = 1.

  return to_T(module_validity_mat)
