import os
import tensorflow as tf
from os.path import join

from basic.main import main as m
from snmn.config import (cfg, merge_cfg_from_file)
flags = tf.app.flags

# Names and directories
flags.DEFINE_string("model_name", "NMN", "Model name [NMN]")
flags.DEFINE_string("dataset", "hotpotqa", "[hotpotqa]")
flags.DEFINE_string("data_dir", "data/hotpotqa", "Data dir")
flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("answer_path", "", "Answer path []")
flags.DEFINE_string("eval_path", "", "Eval path []")
flags.DEFINE_string("load_path", "", "Load path []")
flags.DEFINE_string("shared_path", "", "Shared path []")

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
flags.DEFINE_bool("eval", True, "eval? [True]")
flags.DEFINE_bool("wy", False, "Use wy for loss / eval? [False]")
flags.DEFINE_bool("na", False, "Enable no answer strategy and learn bias? [False]")
flags.DEFINE_float("th", 0.5, "Threshold [0.5]")
flags.DEFINE_integer("gradient_accum_steps", 1, ".")

# Training / test parameters
flags.DEFINE_string("reasoning_layer", None, "[NONE | snmn]")
flags.DEFINE_integer("batch_size", 16, "Batch size [16]")
flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 80000, "Number of steps [20000]")
flags.DEFINE_integer("load_step", 0, "load step [0]")
flags.DEFINE_float("init_lr", 0.001, "Initial learning rate [0.001]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("highway_keep_prob", 1.0, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 80, "Hidden size [100]") #100
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]") #100
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")
flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")
flags.DEFINE_integer("emb_dim", 300, ".")

# Optimizations
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", False, "Length optimization? [False]")
flags.DEFINE_bool("cpu_opt", False, "CPU optimization? GPU computation can be slower [False]")
flags.DEFINE_string("optimizer", "Adam", "[Adam | SGD]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 2250, "sent size th [2250]")  # Since we are concatenating sentences together, the sent_size_th is the same as para_size_th.
flags.DEFINE_integer("num_sents_th", 1, "num sents th [1]")
flags.DEFINE_integer("ques_size_th", 80, "ques size th [80]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 2250, "para size th [2250]")

# Advanced training options
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")

# Ablation options
flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")

# Training options
flags.DEFINE_bool("supervise_bridge_entity", False, ".")
flags.DEFINE_float("attn_loss_coeff", 0.5, ".")
flags.DEFINE_float("first_attn_loss_coeff", 0.5, ".")
flags.DEFINE_bool("self_att", True, ".")
flags.DEFINE_bool("cudnn_rnn", True, ".")

# Decoding options
flags.DEFINE_string("compute_em_f1_on", "phrase", "[phrase | span]")

############################ NMN for HotpotQA ###########################################
flags.DEFINE_string("nmn_attention_type", 'conv', ".")
flags.DEFINE_boolean("nmn_cfg", True, ".")
flags.DEFINE_float("hop0_attn_loss_coeff", 0.5, ".")
flags.DEFINE_float("yesno_loss_coeff", 1.0, ".")
flags.DEFINE_string("nmn_qtype_class", "ctrl_st", "[mem_last | ctrl_st]")
flags.DEFINE_string("nmn_span_pred_hop", 'second', '[last | second]')
flags.DEFINE_string("nmn_mem_init", "random", "[zero | random]")
flags.DEFINE_boolean("nmn_dropout", False, ".")

## Controller ##
flags.DEFINE_boolean("nmn_separate_controllers", True, ".")
flags.DEFINE_boolean("train_nmn_ctrl_separately", True, ".")
flags.DEFINE_boolean("occasional_train_nmn_ctrl", True, '.')
flags.DEFINE_integer("train_nmn_ctrl_period", 500, ".")
flags.DEFINE_integer("train_nmn_ctrl_steps", 300, ".")
flags.DEFINE_string("train_nmn_ctrl_source", 'dev', '[dev | train]')

## Transform Module ##
flags.DEFINE_integer("nmn_relocate_type", 1, ".")
flags.DEFINE_boolean("nmn_relocate_move_ptr", False, ".")

## Compare Module ##
flags.DEFINE_boolean("nmn_compare_mlp", False, ".")
flags.DEFINE_boolean("nmn_yesno_concat_c_last", True, ".")
flags.DEFINE_integer("nmn_compare_fun", 1, ".")

flags.DEFINE_string("out_dir", "", "output directory.")
flags.DEFINE_string("save_dir", "", "output directory.")
flags.DEFINE_string("my_log_dir", "", "output directory.")
flags.DEFINE_string("eval_dir", "", "output directory.")
flags.DEFINE_string("answer_dir", "", "output directory.")

flags.DEFINE_integer("max_num_sents", 0, "As name.")
flags.DEFINE_integer("max_sent_size", 0, "As name.")
flags.DEFINE_integer("max_para_size", 0, "As name.")
flags.DEFINE_integer("max_ques_size", 0, "As name.")
flags.DEFINE_integer("max_ques_sub_size", 0, "As name.")
flags.DEFINE_integer("max_word_size", 0, "As name.")
flags.DEFINE_integer("char_vocab_size", 0, "As name.")
flags.DEFINE_integer("word_emb_size", 0, "As name.")
flags.DEFINE_integer("word_vocab_size", 0, "As name.")

def main(_):
  config = flags.FLAGS
  
  if config.nmn_cfg:
    cfg_path = os.path.join("snmn/cfgs", config.run_id+'.yaml')
    merge_cfg_from_file(cfg_path)

  config.data_dir = os.path.join('data', config.dataset)

  if config.mode == 'test':
    config.input_keep_prob = 1.0
    config.highway_keep_prob = 1.0

  config.out_dir = os.path.join(config.out_base_dir, config.dataset, config.model_name, str(config.run_id).zfill(2))
  
  if config.dataset == 'hotpotqa':
    if config.emb_dim == 300:
      config.data_dir = join(config.data_dir, '840b300d')
    elif config.emb_dim == 100:
      config.data_dir = join(config.data_dir, '6b100d')
    else:
      raise NotImplementedError
  #if config.supervise_bridge_entity:
  config.data_dir += '-bridge'

  m(config)

if __name__ == "__main__":
  tf.app.run()
