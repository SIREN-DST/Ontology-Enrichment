import pickle, os, configparser
import tensorflow_hub as hub
import numpy as np
import concurrent.futures

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

def check_field(section, key, key_name, optional=False):
    ''' Checks config.ini for existence of config[section][key], and also whether
     that refers to a file that exists. Prints a Warning or Error accordingly
     Args:
    - section: Refers to a section in `config.ini`
    - key: Refers to a key under section in `config.ini`
    - key_name: Refers to the field being queried. Used to generate error msg if needed
    - optional: Signifies whether presence of key in `config.ini` is optional or not
      '''
    try:
        field = os.path.abspath(config[section][key])
    except:
        if optional:
            print ("WARNING: No " + key_name + " specified in config.ini")
            return
        else:
            raise KeyError("ERROR: No " + key_name + " specified. Check config.ini")

    if os.path.exists(field):
        return field
    else:
        if optional:
            print ("WARNING: No file found by the name of", config[section][key])
            return
        else:
            raise FileNotFoundError("No file found by the name of", config[section][key])

def compare_similarity(arg):
    '''Finds most similar word for `word_str` among subset of corpus words being compared'''
    word_string, word_embed = arg
    max_sim = -1000
    closest_word = ""
    for emb in corpus_embeddings:
        sim = np.dot(word, emb[1])
        if sim > max_sim:
            max_sim = sim
            closest_word = emb[0]
    return (word_string, closest_word, max_sim)

def entity_to_id(db, entity, resolve=True):
    ''' Lookup db for entity ID. Fills suc '''
    global success, failed
    entity_id = db.get(entity)
    if entity_id:
        success.append(entity)
        return
    failed.append(entity)
    return

def extractUSEEmbeddings(words):
    embed = hub.KerasLayer(USE_folder)
    word_embeddings = embed(words)
    return word_embeddings.numpy()

corpus_embeddings, failed, success = [], [], []

def run():
    global corpus_embeddings

    train_file = check_field('dataset', 'train_file', "training dataset")

    word2id_db = load_db(check_field('preprocessing', 'word2id_db', "Word-to-id database"))
    resolved_db = check_field('preprocessing', 'resolved_file', "Resolved file", True)
    words = list(word2id_db.keys())

    train_dataset = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(train_file).read().split("\n")}
    _ = [[entity_to_id(word2id_db, elem) for elem in tup] for tup in train_dataset.keys()]
    failed_embeds = extractUSEEmbeddings(failed)
    
    print ("Extracted failed embeddings")
    results = {i: ("", -1000) for i in failed}

    len_part = 10000
    n_parts = ceil(len(words)/len_part)
    closest_word = ""

    for i in range(n_parts):
        words_part = words[i*len_part:(i+1)*len_part]
        corpus_embeddings = list(zip(words_part, extractUSEEmbeddings(words_part)))
        output = {}
        args = [(word, failed_embeds[i]) for i,word in enumerate(failed)]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(compare_similarity, args):
                output[result[0]] = (result[1], result[2])
            executor.shutdown(wait=True)
        results = {i: results[i] if results[i][1] > output[i][1] else output[i] for i in results}

    pickle.dump(results, open(resolved_db, "wb"))

if __name__ == '__main__':
    run()
