import pickle, os, configparser, torch, random
import numpy as np
from math import ceil
from itertools import count
from collections import defaultdict
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score
from model import *
from helper import *

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

def id_to_entity(db, entity_id):
    ''' Lookup db for entity using ID '''
    entity = db[str(entity_id)]
    return entity

def id_to_path(db, entity_id):
    ''' Lookup db for path using ID '''
    entity = db[str(entity_id)]
    entity = "/".join(["*##*".join(e.split("_", 1)) for e in entity.split("/")])
    return entity

def entity_to_id(db, entity):
    ''' Lookup db for entity ID. In case of missing word, 
    if `resolve_db` is present, use it otherwise return None '''
    global success, failed
    entity_id = db.get(entity)
    if entity_id:
        success.append(entity)
        return int(entity_id)
    if not resolved_db:
        return -1
    closest_entity = resolved_db.get(entity, "")
    if closest_entity and closest_entity[0] and float(closest_entity[1]) > resolve_threshold:
        success.append(entity)
        return int(db.get(closest_entity[0], -1))
    failed.append(entity)
    return -1

def parse_path(path):
    '''Parses a path by: 
    1. Serializing it by converting into a sequence of edges.
    2. Indexing word, POS, dependency and direction tags to represent edge as 4-tuple'''
    parsed_path = []
    for edge in path.split("*##*"):
        direction, edge = extract_direction(edge)
        if edge.split("/"):
            try:
                embedding, pos, dependency = tuple([a[::-1] for a in edge[::-1].split("/",2)][::-1])
            except:
                print (edge, path)
                raise
            emb_idx, pos_idx, dep_idx, dir_idx = emb_indexer[embedding], pos_indexer[pos], dep_indexer[dependency], dir_indexer[direction]
            parsed_path.append(tuple([emb_idx, pos_idx, dep_idx, dir_idx]))
        else:
            return None
    return tuple(parsed_path)

def parse_tuple(tup):
    '''Extracts paths between a pair of entities (both X->Y and Y->X)'''
    global word2id_db
    x, y = [entity_to_id(word2id_db, elem) for elem in tup]
    paths_x, paths_y = list(extract_paths(relations_db,x,y).items()), list(extract_paths(relations_db,y,x).items())
    path_count_dict_x = { id_to_path(id2path_db, path).replace("X/", tup[0]+"/").replace("Y/", tup[1]+"/") : freq for (path, freq) in paths_x }
    path_count_dict_y = { id_to_path(id2path_db, path).replace("Y/", tup[0]+"/").replace("X/", tup[1]+"/") : freq for (path, freq) in paths_y }
    path_count_dict = {**path_count_dict_x, **path_count_dict_y}
    return path_count_dict

def parse_dataset(dataset):
    '''Main function used to parse dataset. For every pair of entity, it returns
    a) the (serialized and indexed) paths between them 
    b) the count (or frequency of occurence) of each of these paths
    c) the target label'''
    parsed_dicts = [parse_tuple(tup) for tup in dataset.keys()]
    parsed_dicts = [{ parse_path(path) : path_count_dict[path] for path in path_count_dict } for path_count_dict in parsed_dicts]
    paths = [{ path : path_count_dict[path] for path in path_count_dict if path} for path_count_dict in parsed_dicts]
    paths = [{NULL_PATH: 1} if not path_list else path_list for i, path_list in enumerate(paths)]
    counts = [list(path_dict.values()) for path_dict in paths]
    paths = [list(path_dict.keys()) for path_dict in paths]
    targets = [rel_indexer[relation] for relation in dataset.values()]
    return list(to_list_mixed(paths)), counts, targets

# Domain of ontology. Used for naming purposes
domain = check_field(config, "DEFAULT", "domain", "domain name")
output_folder = check_field(config, 'DEFAULT', 'output_folder', "Output Folder", False, True)

# Datasets
train_file = check_field(config, 'dataset', 'train_file', "training dataset", False, True)
test_file = check_field(config, 'dataset', 'test_file', "DBPedia testing dataset", True, True)
knocked_file = check_field(config, 'dataset', 'test_knocked', "Knocked-out dataset", True, True)


# Preprocessing 
word2id_db = load_db(check_field(config, 'preprocessing', 'word2id_db', "Word-to-id database", False, True))
id2word_db = load_db(check_field(config, 'preprocessing', 'id2word_db', "Id-to-word database", False, True))
path2id_db = load_db(check_field(config, 'preprocessing', 'path2id_db', "Path-to-id database", False, True))
id2path_db = load_db(check_field(config, 'preprocessing', 'id2path_db', "Id-to-path database", False, True))
relations_db = load_db(check_field(config, 'preprocessing', 'relations_db', "Relations database", False, True))
resolved_db = load_db(check_field(config, 'preprocessing', 'resolved_file', "Resolved file", True, True), False)

# Parameters
resolve_threshold = float(check_field(config, 'parameters', 'resolve_threshold', "resolve threshold"))
emb_dropout = float(check_field(config, 'parameters', 'emb_dropout', "Embedding layer dropout"))
hidden_dropout = float(check_field(config, 'parameters', 'hidden_dropout', "Hidden layer dropout"))
NUM_LAYERS = int(check_field(config, 'parameters', 'NUM_LAYERS', "Number of LSTM layers"))
HIDDEN_DIM = int(check_field(config, 'parameters', 'HIDDEN_DIM', "Hidden dimension"))
LAYER1_DIM = int(check_field(config, 'parameters', 'LAYER1_DIM', "Layer 1 Output dimension"))
lr = float(check_field(config, 'parameters', 'lr', "Learning rate"))
num_epochs = int(check_field(config, 'parameters', 'epochs', "Number of epochs"))
weight_decay = float(check_field(config, 'parameters', 'weight_decay', "Weight Decay"))
batch_size = int(check_field(config, 'parameters', 'batch_size', "Batch size"))

model_file = output_folder + domain + "_model.pt"
indexers_file = output_folder + domain + "_indexers.pkl"
output_file_prefix = output_folder + domain + "_"

failed, success = [], []

emb_indexer, pos_indexer, dep_indexer, dir_indexer = [defaultdict(count(0).__next__) for i in range(4)]
unk_emb, unk_pos, unk_dep, unk_dir = emb_indexer["<UNK>"], pos_indexer["<UNK>"], dep_indexer["<UNK>"], dir_indexer["<UNK>"]
rel_indexer = {key: idx for (idx,key) in enumerate(relations)}

train_dataset = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(train_file).read().split("\n")}
paths_train, counts_train, targets_train = parse_dataset(train_dataset)
nodes_train = [[emb_indexer[tup[0]], emb_indexer[tup[1]]] for tup in train_dataset]

if test_file:
	test_dataset = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(test_file).read().split("\n")}
	paths_test, counts_test, targets_test  = parse_dataset(test_dataset)
	nodes_test = [[emb_indexer[tup[0]], emb_indexer[tup[1]]] for tup in test_dataset]
else:
	test_dataset = {}
	paths_test, counts_test, targets_test, nodes_test = [], [], [], []

if knocked_file:
	test_knocked = {tuple(l.split("\t")[:2]): l.split("\t")[2] for l in open(knocked_file).read().split("\n")}
	paths_knocked, counts_knocked, targets_knocked = parse_dataset(test_knocked)
	nodes_knocked = [[emb_indexer[tup[0]], emb_indexer[tup[1]]] for tup in test_knocked]
else:
	test_knocked = {}
	paths_knocked, counts_knocked, targets_knocked, nodes_knocked = [], [], [], []

emb_indexer_inv = {emb_indexer[key]: key for key in emb_indexer}
embeds = extractUSEEmbeddings(list(emb_indexer.keys())[1:])
emb_vals = np.array(np.zeros((1, embeds.shape[1])).tolist() + embeds.tolist())

rel_indexer_inv = {rel_indexer[key]: key for key in rel_indexer}

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

def calculate_recall(true, pred):
    ''' Calculates recall of enrichment process by finding 
    accuracy score of relations with non-null labels'''
    true_f, pred_f = [], []
    for i,elem in enumerate(true):
        if elem!=4:
            true_f.append(elem)
            pred_f.append(pred[i])
    return accuracy_score(true_f, pred_f)

def calculate_precision(true, pred):
    ''' Calculates precision of enrichment process by finding 
    accuracy score of relations with non-null labels'''	
    true_f, pred_f = [], []
    for i,elem in enumerate(pred):
        if elem!=4:
            pred_f.append(elem)
            true_f.append(true[i])
    return accuracy_score(true_f, pred_f)

NUM_RELATIONS = len(rel_indexer)
NULL_EDGE = [0, 0, 0, 0]

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

flatten = lambda l: [item for sublist in l for item in sublist]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = OntoEnricher(emb_vals).to(device)
criterion = nn.NLLLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

for epoch in range(num_epochs):
    all_losses = []    
    all_inp = list(zip(nodes_train, paths_train, counts_train, targets_train))
    all_inp_shuffled = random.sample(all_inp, len(all_inp))
    nodes_train, paths_train, counts_train, targets_train = list(zip(*all_inp_shuffled))

    num_edges_all = [[len(path) for path in element] for element in paths_train]
    max_edges = max(flatten(num_edges_all))
    max_paths = max([len(elem) for elem in counts_train])

    dataset_size = len(nodes_train)
    batch_size = min(batch_size, dataset_size)
    num_batches = int(ceil(dataset_size/batch_size))

    for batch_idx in range(num_batches):
        
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx+1) * batch_size
        
        nodes = torch.LongTensor(nodes_train[batch_start:batch_end]).to(device)
        paths = torch.LongTensor(pad_paths(paths_train[batch_start:batch_end], max_paths, max_edges)).to(device)
        counts = torch.DoubleTensor(pad_counts(counts_train[batch_start:batch_end], max_paths)).to(device)
        edgecounts = torch.LongTensor(pad_edgecounts(num_edges_all[batch_start:batch_end], max_paths)).to(device)
        targets = torch.LongTensor(targets_train[batch_start:batch_end]).to(device)
        
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()

        # Run the forward pass
        outputs = model(nodes, paths, counts, edgecounts, max_paths, max_edges)

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())
    
    print("Epoch: {}/{} Mean Loss: {}".format(epoch, num_epochs, np.mean(all_losses)))  

print("Training Complete!")

model_dict = model.state_dict()
model_dict = {key: model_dict[key] for key in model_dict if key!="name_embeddings.weight"}
torch.save(model_dict, model_file)
pickle.dump([pos_indexer, dep_indexer, dir_indexer], open(indexers_file, "wb"))

def test(nodes_test, paths_test, counts_test, targets_test, message):
    predictedLabels, trueLabels = [], []
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
        targets = torch.LongTensor(targets_test[batch_start:batch_end])
        
        outputs = model(nodes, paths, counts, edgecounts, max_paths, max_edges)
        _, predicted = torch.max(outputs, 1)
        predicted = [el.item() for el in predicted]
        targets = [el.item() for el in targets]
        predictedLabels.extend(predicted)
        trueLabels.extend(targets)
        results.extend(["\t".join(tup) for tup in zip(["\t".join([emb_indexer_inv[tup[0]], emb_indexer_inv[tup[1]]]) for tup in nodes.cpu().numpy()], [rel_indexer_inv[l] for l in predicted], [rel_indexer_inv[l] for l in targets])])

    open(output_file_prefix + message + ".tsv", "w+").write("\n".join(results))
    accuracy = accuracy_score(trueLabels, predictedLabels)
    recall = calculate_recall(trueLabels, predictedLabels)
    precision = calculate_precision(trueLabels, predictedLabels)
    try:
        final_metrics = [accuracy, precision, recall, 2 * (precision * recall/(precision + recall))]
    except ZeroDivisionError:
        final_metrics = [accuracy, precision, recall, 0]
    except:
        raise
    print("Final Results ({}): [{}]".format(message, ", ".join([str(el) for el in final_metrics])))

model.eval()
with torch.no_grad():
    if nodes_test:
    	test(nodes_test, paths_test, counts_test, targets_test, "dbpedia")
    if nodes_knocked:
    	test(nodes_knocked, paths_knocked, counts_knocked, targets_knocked, "knocked-out")
