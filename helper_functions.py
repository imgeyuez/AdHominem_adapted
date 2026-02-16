# -*- coding: utf-8 -*-
import spacy
import textacy.preprocessing as tp
import numpy as np
import os
from tqdm import tqdm
import fasttext
import pandas as pd
from sklearn.utils import shuffle
import random
import json


class Corpus(object):

    """
        Class for data preprocessing (8000 Amazon review pairs)
        Modified to process my patient data
    """

    # def __init__(self, test_split=0.2, T_w=20, D_w=300, vocab_size_token=15000, vocab_size_chr=125):
    def __init__(self, T_w=20, D_w=300, vocab_size_token=15000, vocab_size_chr=125):

        # define Spacy tokenizer
        self.tokenizer = spacy.load('en_core_web_lg')
        
        # load raw data
        # self.data_panda = pd.read_csv('{}'.format(os.path.join('data', 'amazon.csv')), sep='\t')

        # load pre-trained fastText word embedding model
        self.WE_dic = fasttext.load_model(os.path.join('data', 'cc.en.300.bin'))

        # dimension of word embeddings
        self.D_w = D_w
        # maximum words per sentence
        self.T_w = T_w

        # split size of test set
        # self.test_split = test_split

        # train set
        self.docs_L_tr = [] # left document?
        self.docs_R_tr = [] # right document?
        self.labels_tr = [] # same author or not?

        # test set
        self.docs_L_te = []
        self.docs_R_te = []
        self.labels_te = []

        # vocabulary sizes
        self.vocab_size_token = vocab_size_token
        self.vocab_size_chr = vocab_size_chr

        # token/word-based vocabulary
        self.V_w = {'<ZP>': 0,  # zero-padding
                    '<UNK>': 1,  # unknown token
                    '<SOS>': 2,  # start of sentence
                    '<EOS>': 3,  # end of sentence
                    '<SLB>': 4,  # start with line-break
                    '<ELB>': 5,  # end with line-break
                    }
        # character vocabulary
        self.V_c = {'<ZP>': 0,  # zero-padding character
                    '<UNK>': 1,  # "unknown"-character
                    }

        # dictionary with token/character counts
        self.dict_token_counts = {}
        self.dict_chr_counts = {}

        # unique list of most frequent tokens/characters
        self.list_tokens = None
        self.list_characters = None

        # word embedding matrix
        self.E_w = None

        #unique list to keep track of documents for vocabulary generation
        self.seen_sessions = set()
        self.sessions_cache = dict()
    
    # extract docs
    def extract_docs(self):
        with open(r"/home/iyuzu/master_thesis/data/train_pairs_v2.1.json", "r", encoding="UTF-8") as f:
            train_pairs = json.load(f)
        with open(r"/home/iyuzu/master_thesis/data/test_pairs.json", "r", encoding="UTF-8") as f:
            test_pairs = json.load(f)

        # -------------------
        # 1) TEST SET
        # -------------------
        for idx in tqdm(range(len(test_pairs)), desc="preprocess TEST docs"):
            p = test_pairs[idx]

            doc_1_id = p["l_session"]["id"]
            doc_2_id = p["r_session"]["id"]

            if doc_1_id in self.sessions_cache.keys():
                # if it has been processed before, take a copy of the value element 
                doc_1 = self.sessions_cache[doc_1_id].copy()
           
            else:
                doc_1 = " ".join(p["l_session"]["messages"]) #doc on left sid

                # preprocessing and tokenizing
                doc_1 = self.preprocess_doc(doc_1)

                # add special tokens
                doc_1 = self.add_special_tokens_doc(doc_1)

                self.sessions_cache[doc_1_id] = doc_1

            # same again for document 2
            if doc_2_id in self.sessions_cache.keys():
                # if it has been processed before, take a copy of the value element 
                doc_2 = self.sessions_cache[doc_2_id].copy()
           
            else:
                doc_2 = " ".join(p["r_session"]["messages"]) #doc on left side

                # preprocessing and tokenizing
                doc_2 = self.preprocess_doc(doc_2)

                # add special tokens
                doc_2 = self.add_special_tokens_doc(doc_2)

                self.sessions_cache[doc_2_id] = doc_2

            label = p["label"]

            # add doc-pair to TEST set
            self.docs_L_te.append(doc_1)
            self.docs_R_te.append(doc_2)
            self.labels_te.append(label)

        # -------------------
        # 2) TRAIN SET
        # -------------------

        for idx in tqdm(range(len(train_pairs)), desc='preprocess docs'):
            p = train_pairs[idx]

            # check, if the ids of the documents has been processed before 
            # and are stored in session_cache
            doc_1_id = p["l_session"]["id"]
            doc_2_id = p["r_session"]["id"]

            if doc_1_id in self.sessions_cache.keys():
                # if it has been processed before, take a copy of the value element 
                doc_1 = self.sessions_cache[doc_1_id].copy()
           
            else:
                doc_1 = " ".join(p["l_session"]["messages"]) #doc on left side

                # preprocessing and tokenizing
                doc_1 = self.preprocess_doc(doc_1)

                # count tokens/characters in train set
                self.count_tokens_and_characters_by_session(p["l_session"]["id"], doc_1)

                # add special tokens
                doc_1 = self.add_special_tokens_doc(doc_1)

                self.sessions_cache[doc_1_id] = doc_1

            # same again for document 2
            if doc_2_id in self.sessions_cache.keys():
                # if it has been processed before, take a copy of the value element 
                doc_2 = self.sessions_cache[doc_2_id].copy()
           
            else:
                doc_2 = " ".join(p["r_session"]["messages"]) #doc on left side

                # preprocessing and tokenizing
                doc_2 = self.preprocess_doc(doc_2)

                # count tokens/characters in train set
                self.count_tokens_and_characters_by_session(p["r_session"]["id"], doc_2)

                # add special tokens
                doc_2 = self.add_special_tokens_doc(doc_2)

                self.sessions_cache[doc_2_id] = doc_2

            # add doc-pair to train set
            self.docs_L_tr.append(doc_1)
            self.docs_R_tr.append(doc_2)
            self.labels_tr.append(p["label"])

    # pre-process single document
    def preprocess_doc(self, doc):

        # pre-process data
        doc = tp.normalize.normalize_unicode(doc)
        doc = tp.normalize_whitespace(doc)
        doc = tp.normalize_quotation_marks(doc)

        # apply spaCy to tokenize doc
        doc = self.tokenizer(doc)

        # build new sentences for pre-processed doc
        doc_new = []
        for sent in doc.sents:
            sent_new = ''
            for token in sent:
                token = token.text
                token = token.replace('\n', '')
                token = token.replace('\t', '')
                token = token.strip()
                sent_new += token + ' '
            doc_new.append(sent_new[:-1])
        return doc_new

    # function for single document
    def add_special_tokens_doc(self, doc):

        # add <SOS>
        N_w = []
        for i, sent in enumerate(doc):
            tokens = sent.split()
            doc[i] = ['<SOS>'] + tokens
            N_w.append(len(doc[i]))

        # add <EOS> or <ELB> or <SLB>
        doc_new = []
        for i, sent in enumerate(doc):
            # short sentence
            if N_w[i] <= self.T_w - 1:
                tokens = sent + ['<EOS>']
                doc_new.append(' '.join(tokens))
            # long sentence
            else:
                while len(sent) > 1:
                    if len(sent) <= self.T_w - 1:
                        tokens = sent[:self.T_w - 1] + ['<EOS>']
                        doc_new.append(' '.join(tokens))
                    else:
                        tokens = sent[:self.T_w - 1] + ['<ELB>']
                        doc_new.append(' '.join(tokens))
                    sent = ['<SLB>'] + sent[self.T_w - 1:]

        return doc_new
    
    # check if session's vocab was counted already
    def count_tokens_and_characters_by_session(self, session_id, doc):
        # wenn das Dokument bereits processed wurde, müssen nicht erneut die 
        # Tokens und Characters gezählt werden
        if session_id in self.seen_sessions:
            return
        self.seen_sessions.add(session_id)
        self.count_tokens_and_characters(doc)

    def count_tokens_and_characters(self, doc):
        for sent in doc:
            tokens = sent.split()
            for token in tokens:
                for chr in token:
                    if chr not in self.dict_chr_counts:
                        self.dict_chr_counts[chr] = 0
                    self.dict_chr_counts[chr] += 1
                if token not in self.dict_token_counts:
                    self.dict_token_counts[token] = 0
                self.dict_token_counts[token] += 1

    # remove rare tokens and characters
    def remove_rare_tok_chr(self):

        # remove rare token types
        q = sorted(self.dict_token_counts.items(), key=lambda x: x[1], reverse=True)
        self.list_tokens = list(list(zip(*q))[0])[:self.vocab_size_token]

        # remove rare character types
        q = sorted(self.dict_chr_counts.items(), key=lambda x: x[1], reverse=True)
        self.list_characters = list(list(zip(*q))[0])[:self.vocab_size_chr]

    # make word- and character-based vocabularies
    def make_wrd_chr_vocabularies(self):

        # add tokens to vocabulary and assign an integer
        for token in self.list_tokens:
            self.V_w[token] = len(self.V_w)

        # word embedding matrix
        self.E_w = np.zeros(shape=(len(self.V_w), self.D_w), dtype='float32')
        r = np.sqrt(3.0 / self.D_w)
        for token in self.V_w.keys():
            idx = self.V_w[token]
            if token in ['<UNK>', '<SOS>', '<EOS>', '<SLB>', '<ELB>']:
                # initialize special tokens
                self.E_w[idx, :] = np.random.uniform(low=-r, high=r, size=(1, self.D_w))
            else:
                # initialize pre-trained tokens
                self.E_w[idx, :] = self.WE_dic[token]

        for c in self.list_characters:
            self.V_c[c] = len(self.V_c)
