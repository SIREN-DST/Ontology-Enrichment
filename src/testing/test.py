import pickle, sys, os, configparser, itertools, glob
import en_core_web_lg, neuralcoref, torch, random 
from spacy.attrs import ORTH, LEMMA
from collections import defaultdict
from itertools import count
import numpy as np
from math import ceil
from itertools import count
import tensorflow_hub as hub
from preprocessing import *
from model import *

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

def to_list(seq):
    ''' Converts list of tuples to list of lists '''
    for item in seq:
        if isinstance(item, tuple):
            yield list(to_list(item))
        elif isinstance(item, list):
            yield [list(to_list(elem)) for elem in item]
        else:
            yield item

def pad_paths(paths, max_paths, max_edges):
    ''' Pads paths with `NULL_EDGE` to resolve uneven lengths and make a matrix '''
    paths_edgepadded = [[path + [NULL_EDGE for i in range(max_edges-len(path))]
        for path in element]
    for element in paths]
    NULL_PATH = [NULL_EDGE for i in range(max_edges)]
    paths_padded = [element + [NULL_PATH for i in range(max_paths-len(element))] 
        for element in paths_edgepadded]
    return np.array(paths_padded)
        
def pad_counts(counts, max_paths):
    ''' Pads counts of paths with 0 for giving 0 count to paths with only `NULL_EDGE`'''
    return np.array([elem + [0 for i in range(max_paths - len(elem))] for elem in counts])

def pad_edgecounts(edgecounts, max_paths):
    ''' Pads counts of edges with 1 for giving 1 count to `NULL_EDGE` (useful while packing)'''
    return np.array([elem + [1 for i in range(max_paths - len(elem))] for elem in edgecounts])

# Domain of ontology. Used for naming purposes
domain = check_field("DEFAULT", "domain", "domain name")
output_folder = check_field('DEFAULT', 'output_folder', "Output Folder", False, True)

# Datasets
webpage_dir = check_field('dataset', 'webpages_dir', "Webpage directory")

# Filtering parameters
domain_keyword = check_field('filtering', 'domain_keyword', "Domain Keyword")
domain_threshold = float(check_field('filtering', 'domain_threshold', "Domain Threshold"))
inter_threshold = float(check_field('filtering', 'inter_threshold', "Inter Threshold"))

# Preprocessing 
word2id_db = load_db(check_field('preprocessing', 'word2id_db', "Word-to-id database", False, True))
id2word_db = load_db(check_field('preprocessing', 'id2word_db', "Id-to-word database", False, True))
path2id_db = load_db(check_field('preprocessing', 'path2id_db', "Path-to-id database", False, True))
id2path_db = load_db(check_field('preprocessing', 'id2path_db', "Id-to-path database", False, True))
relations_db = load_db(check_field('preprocessing', 'relations_db', "Relations database", False, True))
resolved_db = load_db(check_field('preprocessing', 'resolved_file', "Resolved file", True, True), False)
dynamic_db_creation = False if check_field('preprocessing', 'dynamic_db_creation', "Dynamic DB creation") == "False" else True

nlp = en_core_web_lg.load()
# load NeuralCoref and add it to the pipe of SpaCy's model, for coreference resolution
coref = neuralcoref.NeuralCoref(nlp.vocab)
nlp.add_pipe(coref, name='neuralcoref')
nlp.add_pipe(nlp.create_pipe('sentencizer'), before="parser")
nlp.tokenizer.add_special_case('Inc.', [{ORTH: 'Inc', LEMMA: 'Incorporated'}])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

emb_indexer = defaultdict(count(0).__next__)
unk_emb = emb_indexer["<UNK>"]

test_dataset = []

flatten = lambda l: [item for sublist in l for item in sublist]

POS_DIM = 4
DEP_DIM = 6
DIR_DIM = 3
NUM_RELATIONS = len(rel_indexer)
NULL_EDGE = [0, 0, 0, 0]

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

model_file = output_folder + domain + "_model.pt"
indexers_file = output_folder + domain + "_indexers.pkl"


relations = ["hypernym", "hyponym", "concept", "instance", "none"]
NUM_RELATIONS = len(relations)
rel_indexer = {key: idx for (idx,key) in enumerate(relations)}
rel_indexer_inv = {rel_indexer[key]: key for key in rel_indexer}

pos_indexer, dep_indexer, dir_indexer = pickle.load(open(indexers_file, "rb"))

for file in enumerate(sorted(glob.glob(webpage_dir + "*"))):
    paras = [t.text for t in list(nlp(open(file).read()).sents)]
    paras = [nlp(para)._.coref_resolved.replace("\n", " ").replace("  ", " ") for para in paras]
    instances = [tokenize_string(nlp(para).noun_chunks) for para in paras]
    instances_pairs = []
    for instances_sent in instances:
        instances_pairs.extend(list(set(list(itertools.combinations(instances_sent, 2)))))
    all_lines = [list(pair) for pair in instances_pairs if pair]

    embeds = extractUSEEmbeddings([domain_keyword] + entities)
    emb_indexer_page = dict(zip([domain_keyword] + entities, embeds))

    lines = [(entities[i], cos_sim(elem, embeds[0])) for i,elem in enumerate(embeds[1:])]
    scores_dict = {elem[0]: elem[1]>domain_threshold for elem in lines}

    filtered_lines = []
    for elem in all_lines:
        try:
            if get(elem[0], scores_dict) and get(elem[1], scores_dict) and cos_sim(emb_indexer_page[elem[0]], emb_indexer_page[elem[1]])>inter_threshold:
                filtered_lines.append(tuple(elem))
        except Exception as e:
            print (e)
            print ("Error 2: ", elem)

    paths_test, counts_test, nodes_test = preprocess_test(filtered_lines, dynamic_db_creation, file)

    emb_indexer_inv = {emb_indexer[key]: key for key in emb_indexer}
    embeds = extractUSEEmbeddings(list(emb_indexer.keys())[1:])
    emb_vals = np.array(np.zeros((1, embeds.shape[1])).tolist() + embeds.tolist())

    model = OntoEnricher(emb_vals).to(device)
    model.load_state_dict(torch.load(model_file))

    results = []
    num_edges_all = [[len(path) for path in element] for element in paths_test]
    max_edges = max(flatten(num_edges_all))
    max_paths = max([len(elem) for elem in counts_test])

    dataset_size = len(nodes_test)
    batch_size = min(batch_size, dataset_size)
    num_batches = int(ceil(dataset_size/batch_size))

    for batch_idx in range(num_batches):
        
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size

        nodes = torch.LongTensor(nodes_test[batch_start:batch_end]).to(device)
        paths = torch.LongTensor(pad_paths(paths_test[batch_start:batch_end], max_paths, max_edges)).to(device)
        counts = torch.DoubleTensor(pad_counts(counts_test[batch_start:batch_end], max_paths)).to(device)
        edgecounts = torch.LongTensor(pad_edgecounts(num_edges_all[batch_start:batch_end], max_paths)).to(device)
        
        outputs = model(nodes, paths, counts, edgecounts, max_paths, max_edges)
        _, predicted = torch.max(outputs, 1)
        predicted = [el.item() for el in predicted]
        results.extend(["\t".join(tup) for tup in zip(["\t".join([emb_indexer_inv[tup[0]], emb_indexer_inv[tup[1]]]) for tup in nodes.cpu().numpy()], [rel_indexer_inv[l] for l in predicted])])

    output_file = output_folder + file.split("/")[-1].split(".")[0] + ".tsv"
    open(output_file, "w+").write("\n".join(results))
    
    