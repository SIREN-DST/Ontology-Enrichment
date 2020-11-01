import sys, os, glob
import en_core_web_lg, neuralcoref, itertools
from spacy.attrs import ORTH, LEMMA
from scipy import spatial

sys.path.insert(1, os.path.abspath('../corpus-creation-preprocessing/'))
sys.path.insert(1, os.path.abspath('../training/'))

from corpus_parser import *
from preprocessing import *

def cos_sim(a,b):
    # Returns cosine similarity of two vectors
    return 1 - spatial.distance.cosine(a, b)

def clean_noun_chunk(noun_chunks):
    ''' Cleans noun chunks by removing tokens with certain POS tags and brackets '''
    all_parsed_chunks = []
    filt_tokens = ["DET", "ADV", "PUNCT", "CCONJ"]
    for np in noun_chunks:
        start_index = [i for i,token in enumerate(np) if token.pos_ not in filt_tokens][0]
        np_filt = np[start_index:].text
        if "(" not in np_filt and ")" in np_filt:
            np_filt = np_filt.replace(")", "")
        elif "(" in np_filt and ")" not in np_filt:
            np_filt = np_filt.replace("(", "")
        all_parsed_chunks.append(np_filt)
    return list(set(all_parsed_chunks))

def to_tuple(seq):
    ''' Converts nested lists into nested tuples '''
    for item in seq:
        if isinstance(item, list):
            yield tuple(to_tuple(item))
        else:
            yield item

def to_list(seq):
    ''' Converts nested tuples into nested lists '''
    for item in seq:
        if isinstance(item, tuple):
            yield list(to_list(item))
        else:
            yield item

def tokenize_string(tup):
    ''' Tokenizes string and joins it using space as delimiter'''
    return tuple([" ".join([tok.text for tok in nlp(elem)]) for elem in tup])

def parse_test_tuple(tup):
    '''Extracts paths between a pair of entities (both X->Y and Y->X) 
        using paths extracted from the webpage'''
    paths_x = list(instances_db.get(tokenize_string(tup), {}).items())
    paths_y = list(instances_db.get(tokenize_string(tup[::-1]), {}).items())
    path_count_dict_x = { path.replace("X/", tup[0]+"/").replace("Y/", tup[1]+"/") : freq for (path, freq) in paths_x }
    path_count_dict_y = { path.replace("Y/", tup[0]+"/").replace("X/", tup[1]+"/") : freq for (path, freq) in paths_y }
    path_count_dict = {**path_count_dict_x, **path_count_dict_y}
    return path_count_dict


def parse_test_dataset(dataset):
    ''' Special function to parse test dataset. Differs from `parse_dataset` in that 
    it uses a dynamically created path database, sourced from webpage, to extract paths'''
    parsed_dicts = [parse_test_tuple(tup) for tup in dataset.keys()]
    parsed_dicts = [{ parse_path(path) : path_count_dict[path] for path in path_count_dict } for path_count_dict in parsed_dicts]
    paths = [{ path : path_count_dict[path] for path in path_count_dict if path} for path_count_dict in parsed_dicts]
    paths = [{NULL_PATH: 1} if not path_list else path_list for i, path_list in enumerate(paths)]
    counts = [list(path_dict.values()) for path_dict in paths]
    paths = [list(path_dict.keys()) for path_dict in paths]
    targets = [rel_indexer[relation] for relation in dataset.values()]
    return list(to_list_mixed(paths)), counts, targets

# def preprocess_test(webpage_paths):
#     if webpage_paths:


nlp = en_core_web_lg.load()

# load NeuralCoref and add it to the pipe of SpaCy's model, for coreference resolution
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')
nlp.add_pipe(nlp.create_pipe('sentencizer'), before="parser")
nlp.tokenizer.add_special_case('Inc.', [{ORTH: 'Inc', LEMMA: 'Incorporated'}])

for file in glob.glob()