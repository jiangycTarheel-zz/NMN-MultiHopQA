import tensorflow as tf
from tensorflow import convert_to_tensor as to_T, newaxis as ax
from my.tensorflow.nn import softsel, get_logits


def weighted_biattention(h, u, context_dim, h_mask=None, u_mask=None, u_weights=None, scope=None, reuse=False):
  """
  h: [bsz, num_doc, doc_len, emb_dim]
  h_mask: [bsz, num_doc, doc_len]
  u: [bsz, query_len, emb_dim]
  u_mask: [bsz, query_len]
  u_weights: [bsz, query_len]
  """
  #h, h_mask = tf.squeeze(h, axis=1), tf.squeeze(h_mask, axis=1)
  h_len, u_len = tf.shape(h)[2], tf.shape(u)[1]
  M = tf.shape(h)[1]
  u_aug = tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1])
  if u_weights is not None:
    u_weights_aug = tf.tile(tf.expand_dims(u_weights, 1), [1, M, 1]) 
    u_weights_aug_aug = tf.tile(tf.expand_dims(u_weights_aug, 2), [1, 1, h_len, 1])
  h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, u_len])
  u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, h_len, 1])
  hu_mask = h_mask_aug & u_mask_aug

  with tf.variable_scope(scope or 'hotpot_biattention', reuse=reuse):
    h_dot = tf.squeeze(tf.tile(tf.expand_dims(tf.layers.dense(h, 1), 3), [1, 1, 1, u_len, 1]), axis=-1)
    u_dot = tf.squeeze(tf.tile(tf.expand_dims(tf.layers.dense(u_aug, 1), 2), [1, 1, h_len, 1, 1]), axis=-1)
    dot_scale = tf.get_variable("dot_scale", [context_dim])
    cross_dot = tf.einsum('ijkl,ijml->ijkm', h * dot_scale, u_aug)
    att = h_dot + u_dot + cross_dot - 1e30 * (1. - tf.cast(hu_mask, 'float32'))
    weight_one = tf.nn.softmax(att) # [bsz, M, h_len, u_len]
    #weight_two = tf.nn.softmax(tf.reduce_max(att, axis=-1)) # [bsz, M, h_len]
    if u_weights is not None:
      weight_two = tf.nn.softmax(tf.einsum('ijkl,ijl->ijk', att, u_weights_aug))
    else:
      weight_two = tf.nn.softmax(tf.reduce_max(att, axis=-1))
    output_one = tf.einsum('ijkl,ijlm->ijkm', weight_one, u_aug) # [bsz, M, h_len, emb_dim]
    output_two = tf.einsum('ijk,ijkl->ijl', weight_two, h) # [bsz, M, emb_dim]
    output = tf.concat([h, output_one, h * output_one, tf.einsum('ijk,ijlk->ijlk', output_two, output_one)], axis=-1)
    return output, context_dim * 4, weight_one


def zhong_selfatt(U, dim, mask=None, seq_len=None, transform=None, scope=None, reuse=None):
  if mask is None:
    assert seq_len is not None
    mask = tf.expand_dims(tf.sequence_mask(seq_len, tf.shape(U)[1]), axis=1)

  with tf.variable_scope(scope or 'zhong_selfAttention', reuse=reuse):
    W1 = tf.get_variable("W1", [dim, dim])
    b1 = tf.get_variable("b1", [dim,])
    W2 = tf.get_variable("W2", [dim, 1])
    b2 = tf.get_variable("b2", [1,])
    layer1_output = tf.nn.tanh(tf.einsum('ijkl,lt->ijkt', U, W1) + b1)
    logits = tf.nn.tanh(tf.squeeze(tf.einsum('ijkl,lt->ijkt', layer1_output, W2) + b2, axis=-1))
    masked_logits = logits * tf.cast(mask, dtype='float')
    att = tf.nn.softmax(masked_logits)
    output = tf.einsum("ijkl,ijk->ijl", U, att)
    if transform == 'expand':
      output = tf.expand_dims(output, axis=1)
    elif transform == 'squeeze':
      output = tf.squeeze(output, axis=1)
    return output


def hotpot_biattention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
  """
  h: [bsz, num_doc, doc_len, emb_dim]
  h_mask: [bsz, num_doc, doc_len]
  u: [bsz, query_len, emb_dim]
  u_mask: [bsz, query_len]
  """
  #h, h_mask = tf.squeeze(h, axis=1), tf.squeeze(h_mask, axis=1)
  h_len, u_len = tf.shape(h)[2], tf.shape(u)[1]
  M = tf.shape(h)[1]
  u_aug = tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1])
  with tf.variable_scope(scope or 'hotpot_biattention'):
    h_dot = tf.squeeze(tf.tile(tf.expand_dims(tf.layers.dense(h, 1), 3), [1, 1, 1, u_len, 1]), axis=-1)
    u_dot = tf.squeeze(tf.tile(tf.expand_dims(tf.layers.dense(u_aug, 1), 2), [1, 1, h_len, 1, 1]), axis=-1)
    dot_scale = tf.get_variable("dot_scale", [config.hidden_size*2])
    cross_dot = tf.einsum('ijkl,ijml->ijkm', h * dot_scale, u_aug)
    att = h_dot + u_dot + cross_dot - 1e30 * (1. - tf.cast(tf.tile(tf.expand_dims(h_mask, axis=3), [1, 1, 1, u_len]), 'float32'))
    weight_one = tf.nn.softmax(att) # [bsz, M, h_len, u_len]
    weight_two = tf.nn.softmax(tf.reduce_max(att, axis=-1)) # [bsz, M, h_len]
    output_one = tf.einsum('ijkl,ijlm->ijkm', weight_one, u_aug) # [bsz, M, h_len, emb_dim]
    output_two = tf.einsum('ijk,ijkl->ijl', weight_two, h) # [bsz, M, emb_dim]
    output = tf.concat([h, output_one, h * output_one, tf.einsum('ijk,ijlk->ijlk', output_two, output_one)], axis=-1)
    return output


def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
  with tf.variable_scope(scope or "bi_attention"):
    JX = tf.shape(h)[2]
    M = tf.shape(h)[1]
    JQ = tf.shape(u)[1]
    h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1])
    u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1])
    if h_mask is None:
      hu_mask = None
    else:
      h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, JQ])
      u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, JX, 1])
      hu_mask = h_mask_aug & u_mask_aug

    u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, M, JX, JQ]
    u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
    h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
    h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

    if tensor_dict is not None:
      a_u = tf.nn.softmax(u_logits)  # [N, M, JX, JQ]
      a_h = tf.nn.softmax(tf.reduce_max(u_logits, 3))
      tensor_dict['a_u'] = a_u
      tensor_dict['a_h'] = a_h
      variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
      for var in variables:
        tensor_dict[var.name] = var

    return u_a, h_a


def attention_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
  with tf.variable_scope(scope or "attention_layer"):
    JX = tf.shape(h)[2]
    M = tf.shape(h)[1]
    JQ = tf.shape(u)[1]
    if config.q2c_att or config.c2q_att:
      u_a, h_a = bi_attention(config, is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
    if not config.c2q_att:
      u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
    if config.q2c_att:
      p0 = tf.concat(axis=3, values=[h, u_a, h * u_a, h * h_a])
    else:
      p0 = tf.concat(axis=3, values=[h, u_a, h * u_a])
    return p0
