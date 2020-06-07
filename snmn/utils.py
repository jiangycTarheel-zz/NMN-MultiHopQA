import tensorflow as tf
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.nn import softsel, get_logits


def hotpot_biattention(config, is_train, h, u, h_mask, u_mask, indim, scope=None, tensor_dict=None):
  #h, h_mask = tf.squeeze(h, axis=1), tf.squeeze(h_mask, axis=1)
  h_len, u_len = tf.shape(h)[1], tf.shape(u)[1]
  with tf.variable_scope(scope or 'hotpot_biattention'):
    h_dot = tf.squeeze(tf.tile(tf.expand_dims(tf.layers.dense(h, 1), 2), [1, 1, u_len, 1]), axis=-1)
    u_dot = tf.squeeze(tf.tile(tf.expand_dims(tf.layers.dense(u, 1), 1), [1, h_len, 1, 1]), axis=-1)
    dot_scale = tf.get_variable("dot_scale", [indim])
    cross_dot = tf.einsum('ijk,ilk->ijl', h * dot_scale, u)
    att = h_dot + u_dot + cross_dot - 1e30 * (1. - tf.cast(tf.tile(tf.expand_dims(h_mask, axis=2), [1, 1, u_len]), 'float32'))
    weight_one = tf.nn.softmax(att)
    weight_two = tf.nn.softmax(tf.reduce_max(att, axis=-1))
    output_one = tf.einsum('ijk,ikl->ijl', weight_one, u)
    output_two = tf.einsum('ij,ijk->ik', weight_two, h)
    output = tf.concat([h, output_one, h * output_one, tf.einsum('ik,ijk->ijk', output_two, output_one)], axis=-1)
    return output


def biattention_layer(is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
  with tf.variable_scope(scope or "attention_layer"):
    h = tf.expand_dims(h, 1)
    h_mask = tf.expand_dims(h_mask, 1)
    # JX = tf.shape(h)[2]
    # M = tf.shape(h)[1]
    # JQ = tf.shape(u)[1]
    #if config.q2c_att or config.c2q_att:
    u_a, h_a = bi_attention(is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
    # if not config.c2q_att:
    #   u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
    #if config.q2c_att:
    p0 = tf.concat(axis=3, values=[h, u_a, h * u_a, h * h_a])
    #else:
    #  p0 = tf.concat(axis=3, values=[h, u_a, h * u_a])
    return p0


def bi_attention(is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
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

    u_logits = get_logits([h_aug, u_aug], None, True, wd=0., mask=hu_mask,
          is_train=is_train, func='tri_linear', scope='u_logits')  # [N, M, JX, JQ]
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
