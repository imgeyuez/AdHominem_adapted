# -*- coding: utf-8 -*-
from helper_functions import Corpus
import os
import pickle
from datetime import datetime
"""
    - Dataset can be downloaded here: https://github.com/marjanhs/prnn
    - Pretrained word embeddings (binary file): https://fasttext.cc/docs/en/english-vectors.html
    
"""

corpus = Corpus()
corpus.extract_docs()
corpus.remove_rare_tok_chr()
corpus.make_wrd_chr_vocabularies()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

for split, docs_l, docs_r, labels, pair_ids in [
    ("TRAIN", corpus.docs_L_tr, corpus.docs_R_tr,
     corpus.labels_tr, corpus.pair_ids_tr),

    ("DEV", corpus.docs_L_dev, corpus.docs_R_dev,
     corpus.labels_dev, corpus.pair_ids_dev),

    ("TEST", corpus.docs_L_te, corpus.docs_R_te,
     corpus.labels_te, corpus.pair_ids_te),
]:
    assert len(docs_l) == len(docs_r) == len(labels) == len(pair_ids), (
        f"{split}: Length mismatch!"
    )

    assert len(set(pair_ids)) == len(pair_ids), (
        f"{split}: Duplicate pair_ids!"
    )

    print(f"{split}: {len(labels)} pairs")
    
with open(os.path.join("data", f"data_veranda_{timestamp}"), 'wb') as f:
    pickle.dump((
        # TRAIN
        corpus.docs_L_tr,
        corpus.docs_R_tr,
        corpus.labels_tr,
        corpus.pair_ids_tr,

        # DEV
        corpus.docs_L_dev,
        corpus.docs_R_dev,
        corpus.labels_dev,
        corpus.pair_ids_dev,

        # TEST
        corpus.docs_L_te,
        corpus.docs_R_te,
        corpus.labels_te,
        corpus.pair_ids_te,

        # VOCABULARY
        corpus.V_w,
        corpus.E_w,
        corpus.V_c
    ), f)
