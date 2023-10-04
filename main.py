from txtai.embeddings import Embeddings
import torch
import torch.nn as nn
import numpy as np
from itertools import combinations
import operator
from simple_term_menu import TerminalMenu
import csv
import struct
import model

# create uid => title mapping
metadata = {}
with open('metadata.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      metadata[row[0]] = row[3]

# create uid => feature vector mapping
embeddings_map = {}
docs = []

with open('embeddings.bin', 'rb') as embeddings_bin:
  format_str = '8s768f'
  while True:
      bin_data = embeddings_bin.read(struct.calcsize(format_str))
      if not bin_data:
          break

      data = struct.unpack(format_str, bin_data)
      uid = data[0].decode('ascii').strip()
      features = list(data[1:])
      embeddings_map[uid] = features
      docs.append((uid, metadata[uid], features))

embeddings = Embeddings({ "path": "allenai/specter" })
embeddings.index(docs)

def get_embedding(x):
  """
  Get vector representation of x where x can be a (query) string or a document.
  """
  return embeddings.batchtransform([(None, x, None)])[0] if isinstance(x, str) else embeddings.transform(x)

results = {}

def make_input(query_vector, sup_doc_vector, inf_doc_vector):
  """
  Make (query, document-pair) input for model.
  """
  return np.array([query_vector, sup_doc_vector, inf_doc_vector], dtype=np.float32).flatten()

def gen_train_data(query_vector, selected_res = None):
  """
  Generate training data. (We use sup(erior) and inf(erior) to denote relative relevance.)
  """
  pos_train_data, neg_train_data = [], []
  for i, _ in enumerate(results[query]):
      # selected_res is superior to any other result
      if selected_res != None and i != selected_res:
        sup_doc_vector = embeddings_map[results[query][selected_res]]
        inf_doc_vector = embeddings_map[results[query][i]]
        pos_train_data.append(make_input(query_vector, sup_doc_vector, inf_doc_vector))
        neg_train_data.append(make_input(query_vector, inf_doc_vector, sup_doc_vector))
      
      # previous results are superior to current result
      for j in range(i):
        if j == selected_res: continue
        inf_doc_vector = embeddings_map[results[query][i]]
        sup_doc_vector = embeddings_map[results[query][j]]
        pos_train_data.append(make_input(query_vector, sup_doc_vector, inf_doc_vector))
        neg_train_data.append(make_input(query_vector, inf_doc_vector, sup_doc_vector))

      # current result is superior to next results
      for j in range(i+1, len(results[query])):
        if j == selected_res: continue
        sup_doc_vector = embeddings_map[results[query][i]]
        inf_doc_vector = embeddings_map[results[query][j]]
        pos_train_data.append(make_input(query_vector, sup_doc_vector, inf_doc_vector))
        neg_train_data.append(make_input(query_vector, inf_doc_vector, sup_doc_vector))

  pos_train_data = torch.from_numpy(np.array(pos_train_data))
  neg_train_data = torch.from_numpy(np.array(neg_train_data))
  return pos_train_data, neg_train_data

"""
Application loop
"""
while True:

  query = input("QUERY: ")
  query_vector = get_embedding(query)
  if query not in results: 
    # bootstrap model with semantic search results
    results[query] = [x for x, _ in embeddings.search(query, 5)]
    model.train(*gen_train_data(query_vector), 100)

  titles = [metadata[res] for res in results[query]]
  terminal_menu = TerminalMenu(titles)
  selected_res = terminal_menu.show()

  model.train(*gen_train_data(query_vector, selected_res), 10)

  results_combs = list(combinations(results[query], 2))
  results_scores = {}
  for result_pair in results_combs:
    vec1 = embeddings_map[result_pair[0]]
    vec2 = embeddings_map[result_pair[1]]
    is_sup = torch.round(model.model(torch.from_numpy(make_input(query_vector, vec1, vec2)))).item()
    k = result_pair[0]
    results_scores[k] = results_scores.get(k, 0) + is_sup
    k = result_pair[1]
    results_scores[k] = results_scores.get(k, 0) + (1 - is_sup)

  results_scores = dict(sorted(results_scores.items(), key=operator.itemgetter(1), reverse=True))
  results[query] = [k for k, _ in results_scores.items()]
