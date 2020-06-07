import tensorflow as tf
import math
from basic.model import Model
from my.tensorflow import average_gradients
import numpy as np

class Trainer(object):
  def __init__(self, config, model):
    assert isinstance(model, Model)
    self.config = config
    self.model = model
    self.opt = tf.train.AdamOptimizer(config.init_lr)
    self.loss = model.get_loss()
    self.var_list = model.get_var_list()
    self.global_step = model.get_global_step()
    self.summary = model.summary
    self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
    self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

  def get_train_op(self):
    return self.train_op

  def step(self, sess, batch, get_summary=False):
    assert isinstance(sess, tf.Session)
    _, ds = batch
    feed_dict = self.model.get_feed_dict(ds, True)
    
    if get_summary:
      loss, summary, train_op = \
        sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
    else:
      loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
      summary = None
    return loss, summary, train_op


class MultiGPUTrainer(object):
  def __init__(self, config, models):
    model = models[0]
    assert isinstance(model, Model)
    self.config = config
    self.model = model
    self.global_step = model.get_global_step()    
    self.opt = tf.train.AdamOptimizer(config.init_lr)

    if config.train_nmn_ctrl_separately:
      self.var_list = model.get_var_list('nmn')
      self.controller_var_list = model.get_var_list('controller')
      controller_grads_list = []
    else:
      self.var_list = model.get_var_list('all')

    self.summary = model.summary
    self.models = models
    losses, grads_list = [], []

    for gpu_idx, model in enumerate(models):
      with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
        loss = model.get_loss()
        grads = self.opt.compute_gradients(loss, var_list=self.var_list)
        losses.append(loss)
        grads_list.append(grads)
        if config.train_nmn_ctrl_separately:
          controller_grads = self.opt.compute_gradients(loss, var_list=self.controller_var_list)
          controller_grads_list.append(controller_grads)

    self.loss = tf.add_n(losses)/len(losses)
    self.grads = average_gradients(grads_list)
    if config.train_nmn_ctrl_separately:
      self.controller_grads = average_gradients(controller_grads_list)
      controller_grad_vars = [x[1] for x in self.controller_grads]
      controller_gradients = [x[0] for x in self.controller_grads]  
      controller_clipped, _ = tf.clip_by_global_norm(controller_gradients, 2)

      ctrl_accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in self.controller_var_list]
      self.ctrl_zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in ctrl_accum_vars]
      self.ctrl_accum_ops = [ctrl_accum_vars[i].assign_add(gv) for i, gv in enumerate(controller_clipped)]

      if config.gradient_accum_steps == 1:
        self.controller_train_op = self.opt.apply_gradients(zip(controller_clipped, controller_grad_vars), global_step=self.global_step)
      else:
        self.controller_train_op = self.opt.apply_gradients([(ctrl_accum_vars[i], gv[1]) for i, gv in enumerate(self.controller_grads)], global_step=self.global_step)
    
    #self.grads, global_norm = tf.clip_by_global_norm(self.grads, 2)

    grad_vars = [x[1] for x in self.grads]
    gradients = [x[0] for x in self.grads]  
    clipped, _ = tf.clip_by_global_norm(gradients, 2)

    accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in self.var_list]
    self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
    self.accum_ops = [accum_vars[i].assign_add(gv) for i, gv in enumerate(clipped)]
    if config.gradient_accum_steps == 1:
      self.train_op = self.opt.apply_gradients(zip(clipped, grad_vars), global_step=self.global_step)
    else:
      self.train_op = self.opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(self.grads)], global_step=self.global_step)
  
    with tf.control_dependencies([self.train_op]):
      self.dummy = tf.constant(0, name='dummy')
    

  def step(self, sess, batches, get_summary=False, lr=None, train_controller=False, accum_gradients=False):
    config = self.config
    assert isinstance(sess, tf.Session)
    feed_dict = {}
    if config.gradient_accum_steps == 1 or accum_gradients:
      assert batches is not None
      for batch, model in zip(batches, self.models):
        _, ds = batch
        feed_dict.update(model.get_feed_dict(ds, True, sess))

    if accum_gradients:
      accum_ops = self.accum_ops
      if train_controller and config.train_nmn_ctrl_separately:
        accum_ops = self.ctrl_accum_ops

      if get_summary:
        loss, summary, _train_op = \
            sess.run([self.loss, self.summary, accum_ops], feed_dict=feed_dict)
      else:
        loss, _train_op = \
            sess.run([self.loss, accum_ops], feed_dict=feed_dict)
        summary = None
    else:
      train_op = self.train_op
      if train_controller and config.train_nmn_ctrl_separately:
        train_op = self.controller_train_op

      if config.gradient_accum_steps == 1:
        if get_summary:
          loss, summary, _train_op = \
            sess.run([self.loss, self.summary, train_op], feed_dict=feed_dict)
        else:
          loss, _train_op = \
            sess.run([self.loss, train_op], feed_dict=feed_dict)
              
          summary = None
      else:
        _train_op = sess.run(train_op)
        summary, loss = None, 0

    if math.isnan(loss):
      logits, g1, cand_mask, cand_emb = \
        sess.run([self.model.logits, self.model.g1, self.model.cand_mask, self.model.cand_emb], feed_dict)
      print(logits)
      print(candidate_spans[0])
      print(candidate_span_y)
      print("mask: ")
      print(cand_mask[0])
      print("cand_emb: ")
      print(cand_emb[0])
      print(feed_dict[self.model.answer_doc_ids])
      print(feed_dict[self.model.first_doc_ids])
      print(batches[0][1].data['ids'])
      #print(feed_dict[self.model.second_doc_ids])
      exit()
    return loss, summary, _train_op
  