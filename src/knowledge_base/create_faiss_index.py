import pandas as pd
import numpy as np
import faiss

from faiss import write_index, read_index

# usage example
# write_index(index, "large.index")
# index = read_index("large.index")


def create_index(path_df, path_index, normalize=True):

    # get embeddings
    df = pd.read_pickle(path_df)
    embeddings_series = df['passage_embedding']
    np_embeddings = np.array(embeddings_series.to_list()).astype('float32')
    vector_dimension = np_embeddings.shape[1]

    # define index
    # index = faiss.IndexFlatL2(vector_dimension)
    index = faiss.IndexFlatIP(vector_dimension)

    # normalization
    if normalize:
        faiss.normalize_L2(np_embeddings)

    index.add(np_embeddings)
    write_index(index, path_index)




# path_df = '../../knowledge_base_data/multidoc2dial/kb_embeddings_sep_df.csv'
# path_index = '../../knowledge_base_data/multidoc2dial/index_sep.index'

path_df = '../../knowledge_base_data/multidoc2dial/kb_embeddings_df.csv'
path_index = '../../knowledge_base_data/multidoc2dial/index.index'

create_index(path_df, path_index)

# check
index = read_index(path_index)