import sys
sys.path.append('../../AMRSim/preprocess')
sys.path.append('../../AMRSim/sentence-transformers')

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from amr_utils.amr_readers import AMR_Reader
from utils import simplify_amr_nopar
from amrsim_sentence_transformers import ExtendSentenceTransformer, InputExample
from preprocess import generate_ref_edge


import warnings
warnings.filterwarnings('ignore')

reader = AMR_Reader()


def read_amrs_file(file):

    files = [file]
    data = []
    for f in tqdm(files):
        amrs = reader.load(f, remove_wiki=True)
        data.extend(amrs)
    return data



def get_amrs_from_file(src_file):
    src_data = read_amrs_file(src_file)

    amrs_list = []

    for src in tqdm(src_data):
        d = {}
        ref1_sen = ' '.join(src.tokens)
        d['ref1'] = ref1_sen
        ref1_graph = src.graph_string()

        graph_simple, triples = simplify_amr_nopar(ref1_graph)

        d['graph_ref1'] = {}
        graph_simple = ' '.join(graph_simple)
        d['graph_ref1']['amr_simple'] = graph_simple
        d['graph_ref1']['triples'] = json.dumps(triples)

        # print(d) # same as amr below
        amrs_list.append(d)

    return amrs_list


def pooling(embeddings, method):

    if method == 'mean':
        pooled_embeddings = np.mean(embeddings, axis = 0)

    elif method == 'sum':
        pooled_embeddings = np.sum(embeddings, axis = 0)

    elif method == 'max':
        pooled_embeddings = np.max(embeddings, axis = 0)

    return pooled_embeddings





class AMR_EmbeddingsGenerator():

    def __init__(self, pooling_method, device):
        
        sent_model = ExtendSentenceTransformer(model_name_or_path="../../AMRSim/sentence-transformers/output/ct-wiki-bert")
        # self.train = None
        self.model = sent_model.to(device)
        self.tokenizer = self.model.tokenizer
        self.pooling_method = pooling_method
    

    def get_pooling_method(self):
      return self.pooling_method


    def generate_pooled_embedding(self, amrs_list, train):

        max_seq_length = self.model.max_seq_length
        examples = []

        for amr in amrs_list:
            edge_index, edge_type, pos_ids = generate_ref_edge(amr, self.tokenizer, max_seq_length)
            inp_example = InputExample(texts=[amr['graph_ref1']['amr_simple']],
                                        edge_index=edge_index, edge_type=edge_type, pos_ids=pos_ids)

            examples.append(inp_example)

        embeddings_list = []
        # model.training = False
        self.model.training = train

        for example in examples:
            sentences = example.texts[0]
            ref_graphs_index = example.edge_index[0]
            ref_graphs_type = example.edge_type[0]
            ref_pos_ids = example.pos_ids[0]
            embeddings = self.model.encode([sentences], graph_index=[ref_graphs_index],
                                       graph_type=[ref_graphs_type], batch_size=1,
                                       convert_to_numpy=True,
                                       pos_ids=[ref_pos_ids])

            embeddings_list.append(embeddings[0])
        
        
        pooled_embeddings = pooling(np.array(embeddings_list), self.pooling_method)

        return pooled_embeddings
