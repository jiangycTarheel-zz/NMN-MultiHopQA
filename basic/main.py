import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import csv

from basic.evaluator import MultiGPUF1Evaluator
from basic.graph_handler import GraphHandler
from basic.model import get_multi_gpu_models
from basic.trainer import MultiGPUTrainer
from basic.read_data import read_data, get_hotpotqa_data_filter, update_config
from my.tensorflow import get_num_params


def main(config):
  set_dirs(config)
  with tf.device(config.device):
    if config.mode == 'train':
      _train(config)
    elif config.mode == 'test':
      _test(config)
    elif config.mode == 'forward':
      _forward(config)
    else:
      raise ValueError("invalid value for 'mode': {}".format(config.mode))


def set_dirs(config):
  # create directories
  assert config.load or config.mode == 'train', "config.load must be True if not training"
  if not config.load and os.path.exists(config.out_dir):
    raise Exception("no_load is set to True, but the out_dir already exists")
    #shutil.rmtree(config.out_dir)

  config.save_dir = os.path.join(config.out_dir, "save")
  config.my_log_dir = os.path.join(config.out_dir, "log")
  config.eval_dir = os.path.join(config.out_dir, "eval")
  config.answer_dir = os.path.join(config.out_dir, "answer")
  if not os.path.exists(config.out_dir):
    os.makedirs(config.out_dir)
  if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)
  if not os.path.exists(config.my_log_dir):
    os.mkdir(config.my_log_dir)
  if not os.path.exists(config.answer_dir):
    os.mkdir(config.answer_dir)
  if not os.path.exists(config.eval_dir):
    os.mkdir(config.eval_dir)


def _config_debug(config):
  if config.debug:
    config.num_steps = 2
    config.eval_period = 1
    config.log_period = 1
    config.save_period = 1
    config.val_num_batches = 2
    config.test_num_batches = 2


def _train(config):
  if config.dataset == 'hotpotqa':
    data_filter = get_hotpotqa_data_filter(config)
  else:
    raise NotImplementedError

  train_data = read_data(config, 'train', config.load, data_filter=data_filter)
  dev_data = read_data(config, 'dev', True, data_filter=data_filter)
  if config.occasional_train_nmn_ctrl:
    dev_data_partial = read_data(config, config.train_nmn_ctrl_source, True, data_filter=data_filter)
  update_config(config, [train_data, dev_data])

  _config_debug(config)

  word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']
  word2idx_dict = train_data.shared['word2idx']
  idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
  emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
            else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
            for idx in range(config.word_vocab_size)])

  tf.reset_default_graph()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  with sess.as_default():
    models = get_multi_gpu_models(config, emb_mat)
    model = models[0]
    print("num params: {}".format(get_num_params()))
    trainer = MultiGPUTrainer(config, models)
    evaluator = MultiGPUF1Evaluator(config, models, None)
    graph_handler = GraphHandler(config, model)  # controls all tensors and variables in the graph, including loading /saving
    graph_handler.initialize(sess)

  # Begin training
  num_steps = config.num_steps or int(math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
  global_step = 0

  lr = config.init_lr

  for batches in tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus * config.gradient_accum_steps, \
                      num_steps=num_steps, shuffle=True, cluster=config.cluster), total=num_steps):

    if config.gradient_accum_steps == 1:
      INSUFFICIENT_DATA = False
      for batch in batches:
        _, ds = batch
        if len(ds.data['x']) < config.batch_size:
          INSUFFICIENT_DATA = True
          break
      if INSUFFICIENT_DATA:
        continue

      global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
      get_summary = global_step % config.log_period == 0

      loss, summary, train_op = trainer.step(sess, batches, get_summary=get_summary, lr=lr)
        
      if get_summary:
        graph_handler.add_summary(summary, global_step)
      # occasional saving
      if global_step % config.save_period == 0:
        graph_handler.save(sess, global_step=global_step)
      if not config.eval:
        continue

      if config.train_nmn_ctrl_separately and config.occasional_train_nmn_ctrl and global_step % config.train_nmn_ctrl_period == 0:
        num_steps = config.train_nmn_ctrl_steps
        for dev_batches in tqdm(dev_data_partial.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps, shuffle=True), 
                                total=num_steps):
          # Calculate the loss and update the parameter.
          loss, summary, train_op = trainer.step(sess, dev_batches, get_summary=get_summary, lr=lr, train_controller=True)
      
      # Occasional evaluation
      if global_step % config.eval_period == 0:
        num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
        if 0 < config.val_num_batches * config.gradient_accum_steps < num_steps:
          num_steps = config.val_num_batches * config.gradient_accum_steps
        e_dev = evaluator.get_evaluation_from_batches(
          sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))

        graph_handler.add_summaries(e_dev.summaries, global_step)
        e_train = evaluator.get_evaluation_from_batches(
          sess, tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps)
        )
        graph_handler.add_summaries(e_train.summaries, global_step)
        if config.dump_eval:
          graph_handler.dump_eval(e_dev)
        if config.dump_answer:
          graph_handler.dump_answer(e_dev)
    else:
      sess.run(trainer.zero_ops)
      for i in range(config.gradient_accum_steps):
        sub_batches = batches[i*config.num_gpus : (i+1)*config.num_gpus]
        INSUFFICIENT_DATA = False
        for batch in sub_batches:
          _, ds = batch
          if len(ds.data['x']) < config.batch_size:
            INSUFFICIENT_DATA = True
            break
        if INSUFFICIENT_DATA:
          continue

        get_summary = global_step % config.log_period == 0 and i == 0
        loss, summary, train_op = trainer.step(sess, sub_batches, get_summary=get_summary, lr=lr, accum_gradients=True)

        if get_summary:
          graph_handler.add_summary(summary, global_step)
      
      global_step = sess.run(model.global_step) + 1
      _, summary, train_op = trainer.step(sess, None, get_summary=False, lr=lr)

      if global_step % config.save_period == 0:
        graph_handler.save(sess, global_step=global_step)
      if not config.eval:
        continue

      if config.train_nmn_ctrl_separately and config.occasional_train_nmn_ctrl and global_step % config.train_nmn_ctrl_period == 0:
        num_steps = config.train_nmn_ctrl_steps
        for dev_batches in tqdm(dev_data_partial.get_multi_batches(config.batch_size, config.num_gpus * config.gradient_accum_steps, num_steps=num_steps, shuffle=True), 
                                total=num_steps):
          sess.run(trainer.ctrl_zero_ops)
          for i in range(config.gradient_accum_steps):
            sub_dev_batches = dev_batches[i*config.num_gpus : (i+1)*config.num_gpus]
            loss, summary, train_op = trainer.step(sess, sub_dev_batches, get_summary=get_summary, lr=lr, train_controller=True, accum_gradients=True)
          
          _, summary, train_op = trainer.step(sess, None, get_summary=False, lr=lr, train_controller=True)

      # Occasional evaluation
      if global_step % config.eval_period == 0:
        num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
        if 0 < config.val_num_batches * config.gradient_accum_steps < num_steps:
          num_steps = config.val_num_batches * config.gradient_accum_steps
        e_dev = evaluator.get_evaluation_from_batches(
          sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))

        graph_handler.add_summaries(e_dev.summaries, global_step)
        e_train = evaluator.get_evaluation_from_batches(
          sess, tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps)
        )
        graph_handler.add_summaries(e_train.summaries, global_step)
        if config.dump_eval:
          graph_handler.dump_eval(e_dev)
        if config.dump_answer:
          graph_handler.dump_answer(e_dev)

  if global_step % config.save_period != 0:
    graph_handler.save(sess, global_step=global_step)


def _test(config):
  test_data = read_data(config, 'dev', True)
  update_config(config, [test_data])

  _config_debug(config)

  if config.use_glove_for_unk:
    word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
    new_word2idx_dict = test_data.shared['new_word2idx']
    idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
    new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')

  #pprint(config.__flags, indent=2)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  with sess.as_default():
    models = get_multi_gpu_models(config, None)
    model = models[0]
    evaluator = MultiGPUF1Evaluator(config, models, tensor_dict=None)        
    graph_handler = GraphHandler(config, model)

    graph_handler.initialize(sess)
  num_steps = math.ceil(test_data.num_examples / (config.batch_size * config.num_gpus))
  if 0 < config.test_num_batches < num_steps:
    num_steps = config.test_num_batches

  e = None
  
  for multi_batch in tqdm(test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps, cluster=config.cluster), total=num_steps):
    ei = evaluator.get_evaluation(sess, multi_batch)
    e = ei if e is None else e + ei

  print(e)
  
  if config.dump_answer:
    print("dumping answer ...")
    graph_handler.dump_answer(e)
    graph_handler.dump_module_prob(e)

  if config.dump_eval:
    print("dumping eval ...")
    graph_handler.dump_eval(e)


def _get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("config_path")
  return parser.parse_args()


class Config(object):
  def __init__(self, **entries):
    self.__dict__.update(entries)


def _run():
  args = _get_args()
  with open(args.config_path, 'r') as fh:
    config = Config(**json.load(fh))
    main(config)


if __name__ == "__main__":
  _run()
