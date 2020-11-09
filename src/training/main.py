import pickle, os, configparser, torch, random
import numpy as np
from math import ceil
from itertools import count
from collections import defaultdict
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score
from model import *
from preprocessing import *

config = configparser.ConfigParser()
try:
    config.read('config.ini')
except:
    print ("ERROR: No config file. Create a new file called config.ini")
    exit()

def check_field(section, key, key_name, optional=False, ispath=False):
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

# Domain of ontology. Used for naming purposes
domain = check_field("DEFAULT", "domain", "domain name")
output_folder = check_field('DEFAULT', 'output_folder', "Output Folder", False, True)

# Datasets
train_file = check_field('dataset', 'train_file', "training dataset", False, True)
test_file = check_field('dataset', 'test_file', "DBPedia testing dataset", True, True)
knocked_file = check_field('dataset', 'test_knocked', "Knocked-out dataset", True, True)


# Preprocessing 
word2id_db = load_db(check_field('preprocessing', 'word2id_db', "Word-to-id database", False, True))
id2word_db = load_db(check_field('preprocessing', 'id2word_db', "Id-to-word database", False, True))
path2id_db = load_db(check_field('preprocessing', 'path2id_db', "Path-to-id database", False, True))
id2path_db = load_db(check_field('preprocessing', 'id2path_db', "Id-to-path database", False, True))
relations_db = load_db(check_field('preprocessing', 'relations_db', "Relations database", False, True))
resolved_db = load_db(check_field('preprocessing', 'resolved_file', "Resolved file", True, True), False)

# Parameters
resolve_threshold = float(check_field('parameters', 'resolve_threshold', "resolve threshold"))
emb_dropout = float(check_field('parameters', 'emb_dropout', "Embedding layer dropout"))
hidden_dropout = float(check_field('parameters', 'hidden_dropout', "Hidden layer dropout"))
NUM_LAYERS = int(check_field('parameters', 'NUM_LAYERS', "Number of LSTM layers"))
HIDDEN_DIM = int(check_field('parameters', 'HIDDEN_DIM', "Hidden dimension"))
LAYER1_DIM = int(check_field('parameters', 'LAYER1_DIM', "Layer 1 Output dimension"))
lr = float(check_field('parameters', 'lr', "Learning rate"))
num_epochs = int(check_field('parameters', 'epochs', "Number of epochs"))
weight_decay = float(check_field('parameters', 'weight_decay', "Weight Decay"))
batch_size = int(check_field('parameters', 'batch_size', "Batch size"))

model_file = output_folder + domain + "_model.pt"
indexers_file = output_folder + domain + "_indexers.pkl"
output_file_prefix = output_folder + domain + "_"

failed, success = [], []
relations = ["hypernym", "hyponym", "concept", "instance", "none"]
NUM_RELATIONS = len(relations)

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

if test_knocked:
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

POS_DIM = 4
DEP_DIM = 6
DIR_DIM = 3
NUM_RELATIONS = len(rel_indexer)
NULL_EDGE = [0, 0, 0, 0]

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

flatten = lambda l: [item for sublist in l for item in sublist]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = OntoEnricher(emb_vals, pos_indexer, dep_indexer, dir_indexer).to(device)
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
