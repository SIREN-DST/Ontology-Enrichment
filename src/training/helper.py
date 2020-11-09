'''Contains helper functions'''

import pickle, os
import tensorflow_hub as hub

USE_link = "https://tfhub.dev/google/universal-sentence-encoder-large/5?tf-hub-format=compressed"
USE_model = hub.load(USE_link)
NULL_PATH = ((0, 0, 0, 0),)

arrow_heads = {">": "up", "<":"down"}

def extract_paths(db, x, y):
    '''Extract paths between `x` and `y` from `db` and serialize it into a dictionary'''
    key = (str(x) + '###' + str(y))
    try:
        relation = db[key]
        return {int(path_count.split(":")[0]): int(path_count.split(":")[1]) for path_count in relation.split(",")}
    except Exception as e:
        return {}

def extractUSEEmbeddings(words):
    word_embeddings = USE_model(words)
    return word_embeddings.numpy()

def to_list_mixed(seq):
    '''Converts mixed list of tuples into list'''
    for item in seq:
        if isinstance(item, tuple):
            yield list(to_list_mixed(item))
        elif isinstance(item, list):
            yield [list(to_list_mixed(elem)) for elem in item]
        else:
            yield item

def extract_direction(edge):
    '''Converts direction arrow heads into string representation based on positions'''
    if edge[0] == ">" or edge[0] == "<":
        direction = "start_" + arrow_heads[edge[0]]
        edge = edge[1:]
    elif edge[-1] == ">" or edge[-1] == "<":
        direction = "end_" + arrow_heads[edge[-1]]
        edge = edge[:-1]
    else:
        direction = ' '
    return direction, edge

def check_field(config, section, key, key_name, optional=False, ispath=False):
    ''' Checks config.ini for existence of config[section][key], and also whether
     that refers to a file that exists. Prints a Warning or Error accordingly
     Args:
    - section: Refers to a section in `config.ini`
    - key: Refers to a key under section in `config.ini`
    - key_name: Refers to the field being queried. Used to generate error msg if needed
    - optional: Signifies whether presence of key in `config.ini` is optional or not
      '''
    try:
            if not ispath:
                field = config[section][key]
            else:
                field = os.path.abspath(config[section][key])
    except:
            if optional:
                print ("WARNING: No " + key_name + " specified in config.ini")
                return
            else:
                raise KeyError("ERROR: No " + key_name + " specified. Check config.ini")

    if os.path.exists(field) or not ispath:
        return field
    else:
        if optional:
            print ("WARNING: No file found by the name of", config[section][key])
            return
        else:
            raise FileNotFoundError("No file found by the name of", config[section][key])

def preprocess_db(db):
    '''Decodes db keys and values to utf-8'''
    final_db = {}
    for key in db:
        try:
            new_key = key.decode("utf-8")
        except:
            new_key = key
        try:
            new_val = db[key].decode("utf-8")
        except:
            new_val = db[key]
        final_db[new_key] = new_val
    return final_db

def load_db(db_name, encoded=True):
    ''' Loads pickle file. If `encoded`, it also decodes them. '''
    if not db_name:
        return
    return preprocess_db(pickle.load(open(db_name, "rb")))
