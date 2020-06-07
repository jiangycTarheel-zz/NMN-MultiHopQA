import numpy as np
import random

from basic.read_data import DataSet

def get_batch_feed_dict(model, batch, is_train, supervised=True):
  assert isinstance(batch, DataSet)
  config = model.config
  N, M, JX, JQ, VW, VC, d, W = \
    config.batch_size, config.max_num_sents, config.max_sent_size, \
    config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size

  feed_dict = {}

  if config.len_opt:
    """
    Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
    First test without len_opt and make sure no OOM, and use len_opt
    """
    if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
      new_JX = 1
    else:
      new_JX = max(len(sent) for para in batch.data['x'] for sent in para)
    JX = min(JX, new_JX)
    if sum(len(ques) for ques in batch.data['q']) == 0:
      new_JQ = 1
    else:
      new_JQ = max(len(ques) for ques in batch.data['q'])
    JQ = min(JQ, new_JQ)

  if config.cpu_opt:
    if sum(len(para) for para in batch.data['x']) == 0:
      new_M = 1
    else:
      new_M = max(len(para) for para in batch.data['x'])
    M = min(M, new_M)

  x = np.zeros([N, M, JX], dtype='int32')
  cx = np.zeros([N, M, JX, W], dtype='int32')
  x_mask = np.zeros([N, M, JX], dtype='bool')

  q = np.zeros([N, JQ], dtype='int32')
  cq = np.zeros([N, JQ, W], dtype='int32')
  q_mask = np.zeros([N, JQ], dtype='bool')
  # x_group = np.zeros([N], dtype='int32')

  feed_dict[model.x] = x
  feed_dict[model.x_mask] = x_mask
  feed_dict[model.cx] = cx
  feed_dict[model.q] = q
  feed_dict[model.cq] = cq
  feed_dict[model.q_mask] = q_mask
  feed_dict[model.is_train] = is_train
  # feed_dict[model.x_group] = x_group

  if config.use_glove_for_unk:
    feed_dict[model.new_emb_mat] = batch.shared['new_emb_mat']

  X = batch.data['x']
  CX = batch.data['cx']

  if supervised:    
    y = np.zeros([N, M, JX], dtype='bool')
    y2 = np.zeros([N, M, JX], dtype='bool')
    wy = np.zeros([N, M, JX], dtype='bool')

    if config.dataset == 'hotpotqa':
      q_type_labels = np.zeros([N, M], dtype='int32')
      feed_dict[model.q_type_labels] = q_type_labels
      q_yesno_labels = np.zeros([N, M], dtype='int32')
      feed_dict[model.q_yesno_labels] = q_yesno_labels

    na = np.zeros([N], dtype='bool')
    feed_dict[model.na] = na
    if config.dataset == 'hotpotqa':
      yes_no = np.zeros([N], dtype='bool')
      feed_dict[model.yes_no] = yes_no
      for i, yn in enumerate(batch.data['yes_no']):
        yes_no[i] = yn

    if config.supervise_bridge_entity:
      bridge_word_in_context_ids = np.zeros([N, 5], dtype='int32')
      bna = np.zeros([N], dtype='bool')
      feed_dict[model.bridge_word_in_context_ids] = bridge_word_in_context_ids
      feed_dict[model.bridge_na] = bna
      for i, (bridge_word_id, nai, bnai) in enumerate(zip(batch.data['bridge_entity_in_context'], batch.data['na'], batch.data['bridge_ent_na'])):
        bna[i] = bnai
        if nai or bnai:
          continue

        valid_cnt = 0
        for jm, bw_id in enumerate(bridge_word_id):
          if bw_id < JX:
            bridge_word_in_context_ids[i, valid_cnt] = bw_id
            valid_cnt += 1
          if valid_cnt == 5:
            break
            
        if valid_cnt == 0:
          bna[i] = True

    for i, (xi, cxi, yi, nai) in enumerate(zip(X, CX, batch.data['y'], batch.data['na'])):
      if config.dataset == 'hotpotqa':
        q_type_labels[i, 0] = 1 if batch.data['q_type'][i][0] > 0 else 0
        if batch.data['q_type'][i][0] == 2:
          q_yesno_labels[i, 0] = 1

      if nai:
        na[i] = nai
        continue
      start_idx, stop_idx = random.choice(yi)
      j, k = start_idx
      j2, k2 = stop_idx
      if config.single:
        X[i] = [xi[j]]
        CX[i] = [cxi[j]]
        j, j2 = 0, 0
      if config.squash:
        offset = sum(map(len, xi[:j]))
        j, k = 0, k + offset
        offset = sum(map(len, xi[:j2]))
        j2, k2 = 0, k2 + offset
      y[i, j, k] = True
      y2[i, j2, k2-1] = True
      
      if j == j2:
        wy[i, j, k:k2] = True
      else:
        wy[i, j, k:len(batch.data['x'][i][j])] = True
        wy[i, j2, :k2] = True


  def _get_word(word):
    d = batch.shared['word2idx']
    #print(word)
    for each in (word, word.lower(), word.capitalize(), word.upper()):
      if each in d:
        return d[each]
    if config.use_glove_for_unk:
      d2 = batch.shared['new_word2idx']
      for each in (word, word.lower(), word.capitalize(), word.upper()):
        if each in d2:
          return d2[each] + len(d)
    return 1

  def _get_char(char):
    d = batch.shared['char2idx']
    if char in d:
      return d[char]
    return 1

  max_sent_size = 0
  for xi in X:
    for xij in xi:
      max_sent_size = max(max_sent_size, len(xij))

  max_sent_size = min(max_sent_size, config.max_sent_size)

  if config.debug:
    print("max_sent_size: %d" %max_sent_size)

  # Process context words and convert them into word id
  word_count_every_example = []
  for i, xi in enumerate(X):      
    word_count = 0
    if model.config.squash:
      xi = [list(itertools.chain(*xi))]
    for j, xij in enumerate(xi):
      if j == config.max_num_sents:
        raise Exception("Exceed max_num_sents.")
        break

      for k, xijk in enumerate(xij):
        if k == config.max_sent_size:
          break
        each = _get_word(xijk)
        assert isinstance(each, int), each        
        x[i, j, k] = each
        x_mask[i, j, k] = True

        word_count += 1
      # x_group[i] += 1

    word_count_every_example.append(word_count)

  max_word_count = max(word_count_every_example)
  feed_dict[model.max_para_size] = max_word_count
  
  # Process context chars and convert them into char id
  for i, cxi in enumerate(CX):
    word_count = 0
    if model.config.squash:
      cxi = [list(itertools.chain(*cxi))]
    for j, cxij in enumerate(cxi):
      if j == config.max_num_sents:
        raise Exception("Exceed max_num_sents.")
        break

      for k, cxijk in enumerate(cxij):          
        if k == config.max_sent_size:
          break
        
        for l, cxijkl in enumerate(cxijk):
          if l == config.max_word_size:
            break
          cx[i, j, k, l] = _get_char(cxijkl)

        word_count += 1 #

    assert word_count == word_count_every_example[i]

  feed_dict[model.y] = y
  feed_dict[model.y2] = y2
  feed_dict[model.wy] = wy

  for i, qi in enumerate(batch.data['q']):
    if i == 0 and config.debug:
      print(qi)
    for j, qij in enumerate(qi):
      q[i, j] = _get_word(qij)
      q_mask[i, j] = True

  for i, cqi in enumerate(batch.data['cq']):
    for j, cqij in enumerate(cqi):
      for k, cqijk in enumerate(cqij):
        cq[i, j, k] = _get_char(cqijk)
        if k + 1 == config.max_word_size:
          break

  if supervised:
    assert np.sum(~(x_mask | ~wy)) == 0  # if x_mask == 0, then wy must be 0

  num_examples = len(batch.data['x'])  
  feed_dict[model.x] = np.stack(x)
  feed_dict[model.cx] = np.stack(cx)
  feed_dict[model.x_mask] = np.stack(x_mask)

  assert len(x) == len(x_mask)

  return feed_dict
