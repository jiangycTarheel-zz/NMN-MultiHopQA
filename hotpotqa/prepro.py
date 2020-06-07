import argparse
import json
import os
import re
from collections import Counter
from tqdm import tqdm
import string
import numpy as np
from hotpotqa.utils import get_word_span, get_word_idx, process_tokens
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

COMMON_WORDS = list(stopwords.words('english'))
PUNCTUATIONS = list(string.punctuation) + ['–'] + ['—'] + [' '] + ['``'] + ["''"]
SHORT_PUNCT = PUNCTUATIONS.copy() + [' ']
PUNCT_COMMON = PUNCTUATIONS + COMMON_WORDS + [' ']
del SHORT_PUNCT[SHORT_PUNCT.index('+')]

def main():
  args = get_args()
  prepro(args)


def get_args():
  parser = argparse.ArgumentParser()
  source_dir = os.path.join("raw_data", "hotpotqa")
  target_dir = "data/hotpotqa"
  glove_dir = os.path.join("raw_data", "glove")
  parser.add_argument('-s', "--source_dir", default=source_dir)
  parser.add_argument('-t', "--target_dir", default=target_dir)
  parser.add_argument("--train_name", default='train-v1.1.json')
  parser.add_argument('-d', "--debug", action='store_true')
  parser.add_argument("--glove_corpus", default="840B")
  parser.add_argument("--glove_dir", default=glove_dir)
  parser.add_argument("--glove_vec_size", default=300, type=int)
  parser.add_argument("--mode", default="full", type=str)
  parser.add_argument("--single_path", default="", type=str)
  parser.add_argument("--tokenizer", default="PTB", type=str)
  parser.add_argument("--url", default="vision-server2.corp.ai2", type=str)
  parser.add_argument("--port", default=8000, type=int)
  parser.add_argument("--split", default=False, action='store_true')
  parser.add_argument("--suffix", default="")
  parser.add_argument("--find_bridge_entity", default=False, action='store_true')

  args = parser.parse_args()

  if args.glove_vec_size == 300:
    args.target_dir = os.path.join(args.target_dir, '840b300d')
  else:
    args.target_dir = os.path.join(args.target_dir, '6b100d')
  
  if args.find_bridge_entity:
    args.target_dir += '-bridge'

  print(args.target_dir)
  return args

def create_all(args):
  out_path = os.path.join(args.source_dir, "all-v1.1.json")
  if os.path.exists(out_path):
    return
  train_path = os.path.join(args.source_dir, args.train_name)
  train_data = json.load(open(train_path, 'r'))
  dev_path = os.path.join(args.source_dir, args.dev_name)
  dev_data = json.load(open(dev_path, 'r'))
  train_data['data'].extend(dev_data['data'])
  print("dumping all data ...")
  json.dump(train_data, open(out_path, 'w'))


def prepro(args):
  if not os.path.exists(args.target_dir):
    os.makedirs(args.target_dir)
 
  train_in_path = 'hotpot_train_v1.1.json'
  dev_in_path = 'hotpot_dev_distractor_v1.json'

  prepro_each(args, 'dev', out_name='dev', in_path=dev_in_path)
  prepro_each(args, 'train', out_name='train', in_path=train_in_path)


def save(args, data, shared, data_type):
  data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
  shared_path = os.path.join(args.target_dir, "shared_{}.json".format(data_type))
  json.dump(data, open(data_path, 'w'))
  json.dump(shared, open(shared_path, 'w'))


def get_word2vec(args, word_counter):
  glove_path = os.path.join(args.glove_dir, "glove.{}.{}d.txt".format(args.glove_corpus, args.glove_vec_size))
  sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
  total = sizes[args.glove_corpus]
  word2vec_dict = {}
  with open(glove_path, 'r', encoding='utf-8') as fh:
    for line in tqdm(fh, total=total):
      array = line.lstrip().rstrip().split(" ")
      word = array[0]
      vector = list(map(float, array[1:]))
      if word in word_counter:
        word2vec_dict[word] = vector
      elif word.capitalize() in word_counter:
        word2vec_dict[word.capitalize()] = vector
      elif word.lower() in word_counter:
        word2vec_dict[word.lower()] = vector
      elif word.upper() in word_counter:
        word2vec_dict[word.upper()] = vector

  print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), glove_path))
  return word2vec_dict


def compute_answer_span(context, answer, first=True):
  """
  Author: Yichen Jiang
  Find the first occuring answer in the context, and return its span.
  First find independent occurences (' '+answer+' '), if no such span exists, search for answer directly.
  IMPORTANT: After we find an independent occurance, all non-independent answers before it will be ignored because previous context is cut.
  """
  context = context
  _answer = answer
  _answer = re.escape(_answer)

  a = re.search(r'({})'.format(_answer), context)
  if a is None:
    a = re.search(r'({})'.format(_answer.lower()), context.lower())
    if a is None:
      return None, None

  start = a.start()
  end = start + len(answer)
  return start, end


def find_doc_with_answer(start_id, doc_lens):
  total_len = 0
  for i,doc_len in enumerate(doc_lens):
    total_len += doc_len
    if start_id < total_len:
      return i, start_id - (total_len - doc_len)
  assert False, ("Answer not found.")
  return 0, 0


def compute_answer_span_in_doc(split_context, answer, doc_id, lower=False):
  offset = len(' '.join(split_context[:doc_id]))
  _answer = re.escape(answer.lower() if lower else answer)
  a = re.search(r'({})'.format('(?<!([A-Za-z]))'+_answer+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), split_context[doc_id].lower() if lower else split_context[doc_id])
  if a is None:
    a = re.search(r'({})'.format(_answer), split_context[doc_id].lower() if lower else split_context[doc_id])
  assert a is not None, (answer, split_context[doc_id])
  start = a.start()
  
  assert len(split_context[doc_id].lower()) >= len(split_context[doc_id])
  for i in range(min(len(split_context[doc_id].lower()) - len(split_context[doc_id]), 5)):
    if split_context[doc_id][start].lower() != answer[0].lower():
      start -= 1
  start = start + offset 

  if doc_id > 0:
    start += 1
    # +1 because we need to add the whitespace after the document DOC_ID
  end = start + len(answer)
  return start, end


def sort_sp_doc_ids(doc1_id, doc1, doc2_id, doc2, answer, id, alt_doc1=None, alt_doc2=None):
  _answer = re.escape(answer)
  _answer_lower = _answer.lower()
  ans_in_lower = False
  a = re.search(r'({})'.format('(?<!([A-Za-z]))'+_answer+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), doc1)
  b = re.search(r'({})'.format('(?<!([A-Za-z]))'+_answer+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), doc2)
  _pass = False
  if a is None and b is None:
    a = re.search(r'({})'.format('(?<!([A-Za-z]))'+_answer_lower+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), doc1.lower())
    b = re.search(r'({})'.format('(?<!([A-Za-z]))'+_answer_lower+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), doc2.lower())
    if a or b:
      ans_in_lower = True
      _pass = True
    else:
      if alt_doc1 and alt_doc2:
        a = re.search(r'({})'.format('(?<!([A-Za-z]))'+_answer+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), alt_doc1)
        b = re.search(r'({})'.format('(?<!([A-Za-z]))'+_answer+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), alt_doc2)
        if a or b:
          _pass = True
        else:
          a = re.search(r'({})'.format('(?<!([A-Za-z]))'+_answer_lower+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), alt_doc1.lower())
          b = re.search(r'({})'.format('(?<!([A-Za-z]))'+_answer_lower+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), alt_doc2.lower())
          if a or b:
            ans_in_lower = True
            _pass = True
  else:
    _pass = True
  
  if _pass is False:
    a = re.search(r'({})'.format(_answer), doc1)
    b = re.search(r'({})'.format(_answer), doc2)

  if a is None and b is None:
    a = re.search(r'({})'.format(_answer_lower), doc1.lower())
    b = re.search(r'({})'.format(_answer_lower), doc2.lower())
    if a or b:
      ans_in_lower = True
    else:
      if alt_doc1 is None or alt_doc2 is None:
        assert False, (answer, id)
      else:
        a = re.search(r'({})'.format(_answer), alt_doc1)
        b = re.search(r'({})'.format(_answer), alt_doc2)
        if a is None and b is None:
          a = re.search(r'({})'.format(_answer_lower), alt_doc1.lower())
          b = re.search(r'({})'.format(_answer_lower), alt_doc2.lower())
          if a or b:
            ans_in_lower = True
          else:
            assert False, (answer, id)

  if a is None:
    return doc1_id, doc2_id, 1, ans_in_lower
  else:
    return doc2_id, doc1_id, 1 if b is None else 2, ans_in_lower


def find_title_answer_entity(doc1, doc2, title1, title2, answer, word_tokenize, is_bridge=False):
  title2 = word_tokenize(title2)
  entity_list = []
  for token in title2:
    if token.lower() in PUNCTUATIONS + COMMON_WORDS:
      continue
    a = re.search(r'({})'.format('(?<!([A-Za-z]))'+re.escape(token.lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), doc1.lower())
    if a is not None:
      entity_list.append(token)
  
  a = re.search(r'({})'.format('(?<!([A-Za-z]))'+re.escape(answer.lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), doc1.lower())
  if a is not None:
    answer_tok = word_tokenize(answer)
    entity_list.extend([t for t in answer_tok if t not in COMMON_WORDS + PUNCTUATIONS])

  if len(entity_list) == 0 and is_bridge:
    title1 = word_tokenize(title1)
    for token in title1:
      if token.lower() in PUNCTUATIONS + COMMON_WORDS:
        continue
      a1 = re.search(r'({})'.format('(?<!([A-Za-z]))'+re.escape(token.lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), doc2.lower())
      a2 = re.search(r'({})'.format('(?<!([A-Za-z]))'+re.escape(token.lower())+'(?=('+'|'.join([re.escape(t) for t in SHORT_PUNCT])+'))'), doc1.lower())
      if a1 and a2:
        entity_list.append(token)
  return entity_list


def prepro_each(args, data_type, start_ratio=0.0, stop_ratio=1.0, out_name="default", in_path=None):
  if args.tokenizer == "PTB":
    sent_tokenize = nltk.sent_tokenize
    def word_tokenize(tokens):
      return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]
  else:
    raise Exception()

  if not args.split:
    sent_tokenize = lambda para: [para]

  source_path = os.path.join(args.source_dir, in_path) or os.path.join(args.source_dir, "{}.json".format(data_type))
  source_data = json.load(open(source_path, 'r'))

  q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
  q_type = []
  support_docs = []
  na = []  # no answer
  yes_no = []
  cy = []
  x, cx = [], []
  x2 = []
  answers = []
  p, p2 = [], []
  para_lens = []

  word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
  start_ai = int(round(len(source_data) * start_ratio))
  stop_ai = int(round(len(source_data) * stop_ratio))

  if args.find_bridge_entity:
    bridge_entities, bridge_entities_in_context = [], []
    bridge_ent_na = []
 
  for ai, article in enumerate(tqdm(source_data[start_ai:stop_ai])):

    supporting_facts = article['supporting_facts'].copy()
    answer = article['answer'].strip()
    query = article['question']
    all_titles = []

    for si, support in enumerate(article['context']):
      all_titles.append(support[0])

    supports = article['context'].copy()

    qi = word_tokenize(query)
    cqi = [list(qij) for qij in qi]

    xp, cxp = [], []
    xp2 = []
    pp, pp2 = [], []
    sp_doc = []
    para_len = []

    titles = []
    for si, support in enumerate(article['context']):
      titles.append(support[0])
      supports[si] = ''.join(support[1])

    # supporting facts
    sp_facts_div_by_doc = {}
    for sp_doc_title, sp_sent_id in supporting_facts:
      if sp_doc_title in titles:
        sp_doc_id = titles.index(sp_doc_title)
        if len(article['context'][sp_doc_id][1]) <= sp_sent_id:
          print(ai)
          continue
        if sp_doc_id not in sp_doc:
          sp_doc.append(sp_doc_id)
          sp_facts_div_by_doc[sp_doc_id] = article['context'][sp_doc_id][1][sp_sent_id]
        else:
          sp_facts_div_by_doc[sp_doc_id] = sp_facts_div_by_doc[sp_doc_id] + article['context'][sp_doc_id][1][sp_sent_id]
    
    x.append(xp)
    cx.append(cxp)
    p.append(pp)
    x2.append(xp2)
    p2.append(pp2)
    support_docs.append(sp_doc)
    para_lens.append(para_len)

    context = ''
    fst_n_doc_len = 0

    ## To store sent-level support facts
    if 'supporting_facts' in article:
      sp_set = set(list(map(tuple, supporting_facts)))
    else:
      sp_set = set()

    context_tokens, text_context = [], ''
    def _process(sent, is_sup_fact, cur_title, is_title=False):
      nonlocal text_context
      sent_tokens = word_tokenize(sent)
      if is_title:
        sent = '<t> {} </t>'.format(sent)
        sent_tokens = ['<t>'] + sent_tokens + ['</t>']
      N_tokens, my_N_tokens = len(context_tokens), len(sent_tokens)
      text_context += sent
      context_tokens.extend(sent_tokens)
      return sent_tokens

    xi_len, xi_cum_len = [], 0
    for pi, para in enumerate(article['context']):
      cur_title, cur_para = para[0], para[1]
      _context = ''
      cur_title, cur_para = para[0], para[1]

      for sent_id, sent in enumerate(cur_para):
        sent = sent.replace("''", '" ').replace("``", '" ').replace('  ', ' ').replace(' ', ' ')
        is_sup_fact = (cur_title, sent_id) in sp_set
        _process(sent, is_sup_fact, cur_title)
        _context += sent

      if pi < len(article['context']) - 1:
        text_context += ' '
      para_len.append(len(context_tokens))
      pp2.append(_context)
      xi_len.append(len(context_tokens) - xi_cum_len)
      xi_cum_len = len(context_tokens)

    xi = [context_tokens]
    context = text_context
    
    cxi = [[list(xijk) for xijk in xij] for xij in xi]
    xp.append(xi[0])
    cxp.append(cxi[0])
    pp.append(context)
    
    # Only "+= 1" because every sets of support_docs corresponds to only 1 question.
    # In SQuAD, every paragraph can have multiple (len(para['qas'])) questions.
    for xij in xi:  # for sentence in context
      for xijk in xij:  # for word in sentence
        word_counter[xijk] += 1
        lower_word_counter[xijk.lower()] += 1
        for xijkl in xijk:
          char_counter[xijkl] += 1

    # answer
    yi = []
    cyi = []
    q_typei = []
    indoc_yi = []
    answer_text = answer

    doc1_id = sp_doc[0]
    doc2_id = sp_doc[1]
    if answer == '':
      raise Exception("Answer is empty.")
    else:  
      num_sp_w_answer = 0
      if answer == 'yes':
        yi.append([(0, -1), (0, -1)])
        na.append(False)
        yes_no.append(True)
        q_typei.append(1)
      elif answer == 'no':
        yi.append([(0, -2), (0, -2)])
        na.append(False)
        yes_no.append(True)
        q_typei.append(2)
      else:
        q_typei.append(0)
        
        answer_start, answer_stop = compute_answer_span(context, answer) # Find first matching span
        if answer_start is None:
          print(answer)
          print(ai)
          print(context)
          exit()
        else:
          na.append(False)
          yes_no.append(False)

          _pp_alt = pp2
          _pp = sp_facts_div_by_doc
          doc1_id, doc2_id, num_sp_w_answer, ans_in_lower = sort_sp_doc_ids(sp_doc[0], _pp[sp_doc[0]], sp_doc[1], _pp[sp_doc[1]], answer, ai, alt_doc1=_pp_alt[sp_doc[0]], alt_doc2=_pp_alt[sp_doc[1]])
          answer_start, answer_stop = compute_answer_span_in_doc(_pp_alt, answer, doc2_id, lower=ans_in_lower)
          assert ' '.join(_pp_alt) == context, (ai, ' '.join(_pp_alt)[:1000], context[:1000])

          yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)

          assert len(xi[yi0[0]]) > yi0[1]
          assert len(xi[yi1[0]]) >= yi1[1], (len(xi[yi1[0]]), yi1[1], yi1[0], ai, xi[0][yi0[1]:])
          
          yi.append([yi0, yi1])
    
    if args.find_bridge_entity:
      _pp = pp2

      be_word = []
      be_word = find_title_answer_entity(_pp[doc1_id], _pp[doc2_id], titles[doc1_id], titles[doc2_id], answer, nltk.word_tokenize, is_bridge=(num_sp_w_answer==1))

      be_word_id_in_doc, be_word_id_in_context = [], []
      bridge_entities.append(be_word_id_in_doc)
      bridge_entities_in_context.append(be_word_id_in_context)
      if len(be_word) > 0:
        bridge_ent_na.append(False if article['type'] == 'bridge' else True )
        for entity in be_word:
          entity_start, entity_stop = compute_answer_span_in_doc(_pp, entity, doc1_id, lower=True)
          yi0, yi1 = get_word_span(context, xi, entity_start, entity_stop)
          entity_doc_id, entity_id_in_doc = find_doc_with_answer(yi0[1], xi_len)
          be_word_id_in_doc.append(entity_id_in_doc)
          assert entity_doc_id == doc1_id, (entity_doc_id, doc1_id)
          be_word_id_in_context.append(yi0[1])
      else:
        be_word_id_in_doc.append(0)
        be_word_id_in_context.append(0)
        bridge_ent_na.append(True)
    
    for qij in qi:
      word_counter[qij] += 1
      lower_word_counter[qij.lower()] += 1
      for qijk in qij:
        char_counter[qijk] += 1

    q.append(qi)
    cq.append(cqi)
    q_type.append(q_typei)
    y.append(yi)
    ids.append(article['_id'])
    answers.append(answer)

    if args.debug:
      break

  assert len(q) == len(na), (len(qa), len(na))
  assert len(q) == len(y), (len(q), len(y))
  assert len(q) == len(x), (len(q), len(x))

  # Get embedding map according to word_counter.
  word2vec_dict = get_word2vec(args, word_counter)
  lower_word2vec_dict = get_word2vec(args, lower_word_counter)
  
  data = {'q': q, 'cq': cq, 'y': y, 'q_type': q_type, 'ids': ids, 'answers': answers, \
            'na': na, 'x': x, 'cx': cx, 'p': p, 'sp_docs': support_docs, 'yes_no': yes_no, 'para_len': para_lens}
  
  if args.find_bridge_entity:
    data.update({'bridge_entity': bridge_entities, 'bridge_entity_in_context': bridge_entities_in_context, 'bridge_ent_na': bridge_ent_na})
   
  shared = {'word_counter': word_counter, 'char_counter': char_counter, \
    'lower_word_counter': lower_word_counter, 'word2vec': word2vec_dict, \
    'lower_word2vec': lower_word2vec_dict}

  print("saving ...")
  print("no answer: %d" %sum(na))
  save(args, data, shared, out_name)
  
if __name__ == "__main__":
  main()