import sys, os
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

def parse_path(path):
    '''Parses a path by: 
    1. Serializing it by converting into a sequence of edges.
    2. Indexing word, POS, dependency and direction tags to represent edge as 4-tuple
    Overwrites original `parse_path` by using read-only versions of indexers'''
    parsed_path = []
    for edge in path.split("*##*"):
        direction, edge = extract_direction(edge)
        if edge.split("/"):
            try:
                embedding, pos, dependency = tuple([a[::-1] for a in edge[::-1].split("/",2)][::-1])
            except:
                print (edge, path)
                raise
            emb_idx, pos_idx, dep_idx, dir_idx = emb_indexer[embedding], pos_indexer.get(pos, 0), dep_indexer.get(dependency, 0), dir_indexer.get(direction, 0)
            parsed_path.append(tuple([emb_idx, pos_idx, dep_idx, dir_idx]))
        else:
            return None
    return tuple(parsed_path)

def tokenize_string(tup):
    ''' Tokenizes string and joins it using space as delimiter'''
    return tuple([" ".join([tok.text for tok in nlp(elem)]) for elem in tup])

def parse_test_tuple(instances_db, tup):
    '''Extracts paths between a pair of entities (both X->Y and Y->X) 
        using paths extracted from the webpage'''
    paths_x = list(instances_db.get(tokenize_string(tup), {}).items())
    paths_y = list(instances_db.get(tokenize_string(tup[::-1]), {}).items())
    path_count_dict_x = { path.replace("X/", tup[0]+"/").replace("Y/", tup[1]+"/") : freq for (path, freq) in paths_x }
    path_count_dict_y = { path.replace("Y/", tup[0]+"/").replace("X/", tup[1]+"/") : freq for (path, freq) in paths_y }
    path_count_dict = {**path_count_dict_x, **path_count_dict_y}
    return path_count_dict


def parse_dataset_dynamic(dataset, instances_db, resolve=True):
    ''' Special function to parse test dataset. Differs from `parse_dataset` in that 
    it uses a dynamically created path database, sourced from webpage, to extract paths'''
    parsed_dicts = [parse_test_tuple(instances_db, tup) for tup in dataset]
    parsed_dicts = [{ parse_path(path) : path_count_dict[path] for path in path_count_dict } for path_count_dict in parsed_dicts]
    paths = [{ path : path_count_dict[path] for path in path_count_dict if path} for path_count_dict in parsed_dicts]
    paths = [{NULL_PATH: 1} if not path_list else path_list for i, path_list in enumerate(paths)]
    counts = [list(path_dict.values()) for path_dict in paths]
    paths = [list(path_dict.keys()) for path_dict in paths]
    return list(to_list_mixed(paths)), counts

def parse_dataset(dataset, resolve=True):
    '''Main function used to parse test dataset. For every pair of entity, it returns
    a) the (serialized and indexed) paths between them 
    b) the count (or frequency of occurence) of each of these paths'''
    parsed_dicts = [parse_tuple(tup, resolve) for tup in dataset]
    parsed_dicts = [{ parse_path(path) : path_count_dict[path] for path in path_count_dict } for path_count_dict in parsed_dicts]
    paths = [{ path : path_count_dict[path] for path in path_count_dict if path} for path_count_dict in parsed_dicts]
    paths = [{NULL_PATH: 1} if not path_list else path_list for i, path_list in enumerate(paths)]
    counts = [list(path_dict.values()) for path_dict in paths]
    paths = [list(path_dict.keys()) for path_dict in paths]
    return list(to_list_mixed(paths)), counts

def get(key, dictionary):
    try:
        return dictionary[key]
    except KeyboardInterrupt as e:
        sys.exit()
    except:
        print (key)
        dictionary_lower = {elem.lower(): dictionary[elem] for elem in dictionary}
        return dictionary_lower[key.lower()]

def preprocess_test(test_dataset, dynamic_db_creation, file):
    ''' Main function to preprocess test data. `dynamic_db_creation`
    defines whether to use preprocessed relations db or dynamically 
    create a new one by extracting paths from webpage. '''
    resolve = True if resolved_db else False
    if not dynamic_db_creation:
        paths, counts = parse_dataset(test_dataset, resolve)
        nodes = [[emb_indexer[tup[0]], emb_indexer[tup[1]]] for tup in test_dataset]
    else:
        doc = open(file).read()
        all_nounchunks = list(nlp(doc).noun_chunks).copy()

        sentences = [list(nlp(nlp(sent.text)._.coref_resolved.replace("\n", " ").replace("  ", " ")).sents)[0]
                     for sent in nlp(doc).sents]
        # [preprocess(nlp(para).noun_chunks) for para in paras]
        all_deps = []
        instances_db = {}
        for sentence in sentences:
            noun_chunks = [n for n in all_nounchunks if sentence.start <= n.start < n.end - 1 < sentence.end]
            noun_chunks = list(nlp(sentence.text).noun_chunks)
            dependencies = getDependencyPaths(sentence, nlp, noun_chunks, 10)
            for dep in dependencies:
                if not dep:
                    continue
                key = tuple([preprocess_word(nlp(word)) for word in dep[:2]])
                path = "/".join(["*##*".join(e.split("_", 1)) for e in dep[-1].split("/")])
                if key not in instances_db:
                    instances_db[key] = [path]
                else:
                    instances_db[key].append(path)
        instances_db = {key: Counter(instances_db[key]) for key in instances_db}

        paths, counts = parse_dataset_dynamic(test_dataset, instances_db, resolve)
        nodes = [[emb_indexer[tup[0]], emb_indexer[tup[1]]] for tup in test_dataset]
    return paths, counts, nodes
